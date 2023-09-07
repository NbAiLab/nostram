import json
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Set, List, Tuple

from Levenshtein import ratio, editops, matching_blocks

from utils.normalization import normalize
import jsonlines

@dataclass
class Segment:
    file: str
    start: float
    end: float
    text: List[str]

    def to_dict(self) -> Dict:
        return dict(file=self.file, start=self.start, end=self.end, text=self.text)


@dataclass
class Position:
    corp_start: int
    corp_end: int
    seg_start: int
    seg_end: int
    ratio: float
    segment: Segment


@dataclass
class CTMWord:
    start: float
    end: float
    text: str


def load_segments(file: Path) -> List[Segment]:
    ret = []
    with open(file) as f:
        for l in f:
            data = json.loads(l)
            ret.append(Segment(data['file'], data['start'], data['end'], data['text'].strip().split()))
    return sorted(ret, key=lambda x: (x.file, x.start))


@dataclass
class Dictionary:
    word2id: Dict[str, int] = field(default_factory=lambda: {'<unk>': 0})
    id2word: Dict[int, str] = field(default_factory=lambda: {0: '<unk>'})

    def put(self, words: Set):
        for id, word in enumerate(sorted(list(words))):
            self.word2id[word] = id + 1
            self.id2word[id + 1] = word

    def get_id(self, word: str, warn_oov: bool = False) -> int:
        if word not in self.word2id:
            if warn_oov:
                print(f'WARN: missing word "{word}"')
            return 0
        return self.word2id[word]

    def get_word(self, id: int) -> str:
        return self.id2word[id]

    def get_text(self, ids: List[int]) -> str:
        return ' '.join([self.get_word(x) for x in ids])

    def get_ids(self, text: str, warn_oov: bool = False) -> List[int]:
        return self.to_ids(text.strip().split(), warn_oov)

    def to_ids(self, text: List[str], warn_oov: bool = False) -> List[int]:
        return [self.get_id(x, warn_oov) for x in text]

    def to_words(self, ids: List[int]) -> List[str]:
        return [self.get_word(x) for x in ids]

    def save(self, file: Path):
        with open(file, 'w') as f:
            for id, word in self.id2word.items():
                f.write(f'{id} {word}\n')

    def load(self, file: Path):
        self.word2id = {}
        self.id2word = {}
        with open(file) as f:
            for l in f:
                tok = l.strip().split()
                id = int(tok[0])
                word = tok[1]
                self.id2word[id] = word
                self.word2id[word] = id


class Matcher:
    '''Matcher utility class

    This class loads a large text corpus and allows matching short text segments to it.
    '''

    def __init__(self, corpus: Path):
        lines = []
        self.corpus = []
        words = set()
        self.vocab = Dictionary()
        with open(corpus) as f:
            for l in f:
                tok = normalize(l.strip())
                tok = tok.strip().split()
                lines.append(tok)
                words.update(tok)
            self.vocab.put(words)
            for l in lines:
                for w in l:
                    self.corpus.append(self.vocab.get_id(w))

    def _findall(self, id):
        ret = []
        off = 0
        N = len(self.corpus)
        while off < N:
            try:
                pos = self.corpus.index(id, off)
                ret.append(pos)
                off = pos + 1
            except ValueError:
                break
        return ret

    def _close_match(self, ids: List[int]) -> Tuple[int, int]:
        N = len(ids)
        if N == 0:
            return 0, 0
        p1 = self._findall(ids[0])
        poff = 1
        while (len(p1) == 0 or len(p1) > 1000) and poff < N:
            p2 = self._findall(ids[poff])
            p1 = [x - poff for x in p2]
            poff += 1
        max_r = 0
        pf = []
        for p in p1:
            sm = SequenceMatcher(a=ids, b=self.corpus[p:p + N], autojunk=False)
            r = sm.ratio()
            if r > max_r:
                max_r = r
                pf = [p]
            elif r == max_r:
                pf.append(p)

        if len(pf) == 0:
            print('ERROR: no candidates found!')
            return 0, 0
        # elif len(pf) > 1:
        #     print('WARNING: multiple candidates found!')

        pf = pf[0]

        # print('C  ' + self.vocab.get_text(self.corpus[pf:pf + N + 10]))
        # print('S  ' + self.vocab.get_text(ids))

        mb = SequenceMatcher(a=ids, b=self.corpus[pf:pf + N + 10], autojunk=False).get_matching_blocks()
        # print(mb)
        m = mb[-2]
        M = m.b + m.size

        return pf, pf + M

    def _ids2str(self, ids: List[int]) -> str:
        return ''.join([chr(x) for x in ids])

    def match(self, segments: List[Segment]) -> List[Position]:
        positions = []

        # skip segments at start that have too little words
        sit = 0
        for sit in range(len(segments)):
            N = len(segments[sit].text)
            if N >= 15:
                break
            print(f'Skipping {sit}\'th segment because too little words: {N}')
            positions.append(Position(-1, -1, -1, -1, 0, segments[sit]))

        # find position of first (non-trivial) segment
        segint = self.vocab.to_ids(segments[sit].text)

        b, e = self._close_match(segint)
        d = ratio(self._ids2str(self.corpus[b:e]), self._ids2str(segint))
        hb = 0
        he = len(segint)
        positions.append(Position(b, e, hb, he, d, segments[sit]))

        # for other segments in sequence
        for sit in range(sit + 1, len(segments)):
            segint = self.vocab.to_ids(segments[sit].text)

            # determine amount of words to match in reference
            L = len(segint)
            if L < 2:
                # print(f'{sit} too short: {L}')
                positions.append(Position(-1, -1, -1, -1, 0, segments[sit]))
                continue
            if (L * 1.5) < 10:
                L += int(L * 1.5)
            else:
                L += 10
            # extract that portion of reference
            tm = self.corpus[e:e + L]

            # match ASR segment to reference
            ops = editops(self._ids2str(tm), self._ids2str(segint))
            mb = matching_blocks(ops, len(tm), len(segint))

            # this should happen rarely
            if len(mb) < 2:
                d = 0
            else:
                b = e + mb[0][0]
                e = e + mb[-2][0] + mb[-2][2]

                hb = mb[0][1]
                he = mb[-2][1] + mb[-2][2]

                # get the ratio of segments that match in length
                d = ratio(self._ids2str(self.corpus[b:e]), self._ids2str(segint))

            # if ratio is bad
            if d < 0.5:
                # try to find the segment elsewhere
                b, e = self._close_match(segint)
                d = ratio(self._ids2str(self.corpus[b:e]), self._ids2str(segint))
                hb = 0
                he = len(segint)

                # if it's still bad, then simply quit trying
                if d < 0.5:
                    # print(f'{sit} filed to match: {d}')
                    positions.append(Position(-1, -1, -1, -1, 0, segments[sit]))
                    continue

            positions.append(Position(b, e, hb, he, d, segments[sit]))

        return positions

    def print_debug(self, positions: List[Position]):
        print(f'Printing {len(positions)} segments:')
        print('==============')
        for pos in positions:
            words = self.vocab.to_words(self.corpus[pos.corp_start:pos.corp_end])
            words = '\t'.join(words)
            print('CORPUS\t' + words)
            words = pos.segment.text[pos.seg_start:pos.seg_end]
            words = '\t'.join(words)
            print('SEGMENT\t' + words)
            print('File:\t' + str(pos.segment.file))
            print('START time:\t' + str(pos.segment.start))
            print('End time:\t' + str(pos.segment.end))
            print(f'RATIO \t {pos.ratio}')
            print('--------------')

    def getCorpus(self):
        return self.vocab.to_words(self.corpus)

    def jsonoutput(self, positions: List[Position],outputfile):
        jsonfilewriter = open(outputfile, "w+")
        #print(self.corpus[0])
        #exit(-1)
        for pos in positions:
            data={}
            words = self.vocab.to_words(self.corpus[pos.corp_start:pos.corp_end])
            words = ' '.join(words)
            data["text"]=words
            data["soundfile"] = str(pos.segment.file)
            data["soundstart"]= str(pos.segment.start)
            data["soundend"] = str(pos.segment.end)
            data["segstart"] =  str(pos.seg_start)
            data["segend"] = str(pos.seg_end)

            words = pos.segment.text[pos.seg_start:pos.seg_end]
            data["wav2vectext"]= ' '.join(words)
            data["ratio"] = pos.ratio
            data["corp_start"]=pos.corp_start
            data["corp_end"] = pos.corp_end
            #print(data)
            jsonfilewriter.write(json.dumps(data, ensure_ascii=False))
            jsonfilewriter.write("\n")
        jsonfilewriter.close()

    def resultingjson(self, positions: List[Position]):
        #print(self.corpus[0])
        #exit(-1)
        result=[]

        for pos in positions:
            data={}
            wordsprev = self.vocab.to_words(self.corpus[pos.corp_start:pos.corp_end])

            #words=wholearraycap[pos.corp_start:pos.corp_end]


            print("##################")
            print(str( ' '.join(wordsprev)) )
            print("##################")

            data["text"]=wordsprev
            data["soundfile"] = str(pos.segment.file)
            data["soundstart"]= str(pos.segment.start)
            data["soundend"] = str(pos.segment.end)
            data["segstart"] =  str(pos.seg_start)
            data["segend"] = str(pos.seg_end)
            #if pos.ratio > 0.5:
             #   print(f"from {pos.segment.start} to {pos.segment.end} , Ratio: {pos.ratio}")
            words = pos.segment.text[pos.seg_start:pos.seg_end]
            data["corp_start"] = pos.corp_start
            data["corp_end"] = pos.corp_end
            data["wav2vectext"]= ' '.join(words)
            data["ratio"] = pos.ratio
            result.append(json.dumps(data, ensure_ascii=False))
            #print(data)
        return result

    def resegment_positions(self, positions: List[Position], max_time_gap: float = 5.0, max_word_gap: int = 10,
                            max_seg_len: float = 20.0 * 60.0) -> List[Position]:
        ret = []
        for pos in positions:
            if len(ret) == 0:
                ret.append(pos)
                continue
            ls = ret[-1]
            if ls.segment.file != pos.segment.file:
                ret.append(pos)
                continue
            len_time = ls.segment.end - ls.segment.start
            delta_time = pos.segment.start - ls.segment.end
            delta_ref = pos.corp_start - ls.corp_end
            # if the gap is between 0..5 seconds and 0..10 words and current length of segments is less than 20 mins
            # then merge segments
            if 0 <= delta_time <= max_time_gap and 0 <= delta_ref <= max_word_gap and len_time < max_seg_len:
                ret[-1].segment.end = pos.segment.end
                ret[-1].seg_end = pos.seg_end + len(ret[-1].segment.text)
                ret[-1].segment.text.extend(pos.segment.text)
                ret[-1].corp_end = pos.corp_end
            else:
                ret.append(pos)
        return ret

    def make_kaldi_dir(self, positions: List[Position], outdir: Path, wavdir: Path):

        files = {}
        for seg in positions:
            if seg.segment.file not in files:
                files[seg.segment.file] = []
            files[seg.segment.file].append(seg)

        outdir.mkdir(exist_ok=True)

        with open(outdir / 'wav.scp', 'w') as wav_f, \
                open(outdir / 'segments', 'w') as seg_f, \
                open(outdir / 'text', 'w') as text_f:
            for fn, (fid, segs) in enumerate(files.items()):
                wav_f.write(f'reco{fn:03} {(wavdir / fid).absolute()}\n')
                for sn, seg in enumerate(segs):
                    seg_f.write(f'seg{fn:03}-{sn:04} reco{fn:03} {seg.segment.start} {seg.segment.end}\n')
                    text_f.write(
                        f'seg{fn:03}-{sn:04} {self.vocab.get_text(self.corpus[seg.corp_start:seg.corp_end])}\n')


def ctm_to_segments(wavscp: Path, ctm: Path, max_gap: float = 1.0, max_len: float = 20.0) -> List[Segment]:
    scp = {}
    with open(wavscp) as f:
        for l in f:
            tok = l.strip().split()
            scp[tok[0]] = ' '.join(tok[1:])

    words = {}
    with open(ctm) as f:
        for l in f:
            tok = l.strip().split()
            if tok[0] not in words:
                words[tok[0]] = []
            start = float(tok[2])
            dur = float(tok[3])
            text = tok[4]
            words[tok[0]].append(CTMWord(start, start + dur, text))

    segs = []
    for utt, word in words.items():
        file = scp[utt]
        for w in sorted(word, key=lambda x: x.start):
            ls = None
            if len(segs) > 0:
                ls = segs[-1]
            if ls and ls.file != file:
                ls = None
            if ls and w.start - ls.end < max_gap and w.end - ls.start <= max_len:
                ls.end = w.end
                ls.text.append(w.text)
            else:
                ns = Segment(file, w.start, w.end, [w.text])
                segs.append(ns)

    segs = sorted(segs, key=lambda x: (x.file, x.start))

    fixed_segs = [segs[0]]
    for seg in segs[1:]:
        ls = fixed_segs[-1]
        if ls.file == seg.file and seg.end - ls.start < max_len:
            ls.text.extend(seg.text)
            ls.end = seg.end
        else:
            fixed_segs.append(seg)

    for seg in fixed_segs:
        seg.start = round(seg.start, 2)
        seg.end = round(seg.end, 2)

    return fixed_segs
