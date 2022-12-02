import os
import re
import warnings
from dataclasses import dataclass

import langid
import numpy as np
import pandas as pd
from tqdm import tqdm

VTT_SPLIT_PATTERN = re.compile(
        r"\n(\d+)\n(-?\d+):(-?\d\d):(-?\d\d)[.,](-?\d{3}) ?--> ?(-?\d+):(-?\d\d):(-?\d\d)[.,](-?\d{3})")


@dataclass
class VTTLine:
    number: int
    ms_start: int
    ms_end: int
    text: str


@dataclass
class VTTFile:
    name: str
    lines: list[VTTLine]

    @classmethod
    def open(cls, file):
        with open(file) as f:
            text = f.read()

        name = os.path.basename(file)
        lines = []
        splits = re.split(VTT_SPLIT_PATTERN, text)
        for i in range(1, len(splits), 10):
            (number, start_hour, start_min, start_sec, start_ms,
             end_hour, end_min, end_sec, end_ms, text) = splits[i:i + 10]

            if any(s.startswith("-") for s in (start_hour, start_min, start_sec, start_ms,
                                               end_hour, end_min, end_sec, end_ms)):
                warnings.warn(f"Negative values found in timestamp ({file=}, {number=})")
                continue

            number = int(number)
            ms_start = 1000 * (60 * (60 * int(start_hour) + int(start_min)) + int(start_sec)) + int(start_ms)
            ms_end = 1000 * (60 * (60 * int(end_hour) + int(end_min)) + int(end_sec)) + int(end_ms)

            line = VTTLine(number, ms_start, ms_end, text)
            lines.append(line)
        return cls(name, lines)


def make_dataframe(vtt_folder):
    files = os.listdir(vtt_folder)

    it = tqdm(sorted(files), "VTT files")
    with open("lines.tsv", "w") as writer:
        writer.write("\t".join(["program", "sub_type", "number", "ms_start", "ms_end", "text"]) + "\n")
        for file in it:
            try:
                vtt = VTTFile.open(os.path.join(vtt_folder, file))
            except ValueError:
                continue
            program, sub_type = file.replace(".vtt", "").split("_")
            for line in vtt.lines:
                if "|" in line.text:
                    raise ValueError(f"{vtt}, {line}")
                text = line.text.strip().replace("\n", "|")
                writer.write(f"{program}\t{sub_type}\t{line.number}\t{line.ms_start}\t{line.ms_end}\t{text}\n")
            writer.flush()
            it.set_postfix_str(file)


def separate_nor_ttv(df: pd.DataFrame):
    """
    Typically TTV is a superset of NOR, so anything in NOR is a spoken foreign language translated to Norwegian text,
     while TTV is Norwegian OR foreign language to Norwegian text.
    This method subtracts any duplicates from TTV, leaving only the Norwegian->Norwegian parts,
     and returns two separate dataframes.
    Note that NOR will sometimes include Norwegian->Sami as well
    """
    # Some programs have overlapping subtitles in different languages, remove those
    dup_sub_start_end = df.duplicated(subset=["program", "sub_type", "ms_start", "ms_end"], keep=False)
    df.drop(df[dup_sub_start_end].index, inplace=True)

    # Mark duplicates by program+time+text, mark anything except first (NOR since alphabetical) as duplicate
    dup_start = df.duplicated(subset=["program", "ms_start", "text"], keep="first")
    dup_end = df.duplicated(subset=["program", "ms_end", "text"], keep="first")
    # Find duplicates only by program+text (both NOR and TTV)
    dup_text_type = df.duplicated(subset=["program", "sub_type", "text"], keep=False)
    dup_text = df.duplicated(subset=["program", "text"], keep=False)

    # Remove any duplicates by time, remove duplicates by text only (unless sub_type is also duplicate)
    mask = ~(dup_start | dup_end) & ~(dup_text & ~dup_text_type)

    ttv = df[mask]
    ttv = ttv[ttv.sub_type == "ttv"]
    nor = df[df.sub_type == "nor"]  # Keep everything from NOR (foreign language to Norwegian and sometimes Sami)

    return ttv, nor


def remove_many_lines(df: pd.DataFrame):
    """
    Subtitles with very many lines are not very common,
     but when they occur it's very often due to the inclusion of a name (or other unspoken info) on the first line.
    """
    df.drop(df[df.text.str.count(r"\|") >= 2].index, inplace=True)


def remove_italics(df: pd.DataFrame):
    """
    Italics are used liberally to denote things like emphasis, narrators, voices from phones etc.
    Generally they are spoken, and should thus be included.
    One special case is parallel translations from tertiary language to Norwegian and Sami,
     with one language italicized and the other not, on separate lines.
    """
    df.drop(df[df.text.str.fullmatch(r"((^|\|)[^|]+)+(\|<i>[^<>|]+</i>)+")].index, inplace=True)
    df.text = df.text.str.replace("</?i>", " ", regex=True).str.strip()


def remove_inaudible(df: pd.DataFrame):
    """
    Some special cases of patterns in subtitles that are not spoken out loud.
    """
    df.text = df.text.str.replace(
        r"(Nord|Sør)-Sápmi:"
        r"|(Norske? )?tekst(er|ar|ing|):.*"
        r"|(English )?subtitles:.*"
        r"|(Ådåsdåjmadiddje|Ođasredaktevra|Ođashoavda)/nyhetsredaktør:.*"
        r"|\((?!(\d|\d\d|1\d\d|Ap|H|Sp|Frp|SV|R|V|MdG|KrF)\))[^)]*\)"  # Remove parentheses without age/political party
        r"|Opptak av simultanteksting",
        # r"|I forrige episode:",
        "", regex=True, flags=re.IGNORECASE)
    df.text = df.text.str.replace(r"^\W+$", "", regex=True)  # Only non-word characters
    df.drop(df[df.text.str.len() == 0].index, inplace=True)


def remove_splits(df: pd.DataFrame, drop_overlapping=False):
    """
    This method goes through and concatenates any lines that belong together,
     e.g. sentences over two lines, sentences over several consecutive lines in different timestamps
     (continuation is denoted by ending a line and starting the next line with -).
    If two lines have dashes that denotes multiple speakers speaking simultaneously or in rapid succession,
     these can optionally be filtered out using `drop_overlapping`
    (note: generation of dataframe replaces newlines with a pipe for parsing)
    """
    df.text = df.text.str.replace("—", "-")  # Inconsistent usage, just stick to the normal dash

    expects_continuation = False
    overlapping = False
    start_index = None
    string = ""
    delete_mask = np.zeros(len(df), dtype=bool)
    for i, text in enumerate(df["text"]):
        text = text.strip()
        is_continuation = text.startswith("-")
        if is_continuation:
            is_continuation = expects_continuation
            overlapping = overlapping or bool(re.fullmatch(r"-.+(\|-.+)+", text))
            if drop_overlapping and overlapping:
                if expects_continuation:  # Invalidate the whole sequence
                    delete_mask[start_index:i + 1] = True
                elif text.endswith("-"):  # Propagate invalidation forward
                    start_index = i
                    expects_continuation = True
                    delete_mask[i] = True
                else:
                    delete_mask[i] = True
                continue
        overlapping = False  # Stop propagation

        expects_continuation = text.endswith("-")

        if is_continuation:
            string += "|" + text
            if not expects_continuation:
                df.iloc[start_index, 5] = string
                df.iloc[start_index, 4] = df.iloc[i, 4]
                delete_mask[start_index + 1: i + 1] = True
                string = None
        elif expects_continuation:
            start_index = i
            string = text
        else:
            start_index = None
            string = None

    df.drop(df[delete_mask].index, inplace=True)

    # Continued word, e.g. `Det er viktig å treffe folk som har mer for-|nuftige interesser enn de gamle kompisene.`
    df.text = df.text.str.replace(r"-\|(?!-)", "", regex=True)

    # This has very inconsistent usage
    df.drop(df[df.text.str.contains(r"-\|-")].index)

    df.text = df.text.str.replace(r"^-|\-?\|-?", " ", regex=True)
    df.text = df.text.str.strip().replace("  +", " ", regex=True)
    df.drop(df[df.text.isna() | (df.text.str.len() == 0)].index, inplace=True)


def add_languages(df: pd.DataFrame):
    """
    Add column with language predictions.
    Makes a comma separated string of language_code,probability for up to 3 languages
    """

    def predict(s):
        ranks = langid.rank(s)
        codes, scores = zip(*ranks)
        scores = np.array(scores)
        scores = scores - scores.max()
        e = np.exp(scores)
        probs = e / e.sum()
        probs = [probs[0]] + probs[1:][(probs.cumsum() < 0.99)[:-1]].tolist()[:3]
        return ",".join(f"{code},{prob * 100:.3g}" for code, prob in zip(codes, probs))

    df["languages"] = df.text.apply(predict)


def remove_invalid_durations(df: pd.DataFrame):
    """
    Some durations will clearly not make sense, either by being too short (some are even negative) or being too long.
    This filters out some of the more extreme durations based on text length.
    """
    # "The most widely known rule on the speed of interlingual subtitles–“the six-seconds rule”–stipulates that a full
    # two-line subtitle should be displayed for six seconds in order for an average viewer to be able to read it.
    # The six-seconds rule is equivalent to approximately 140–150 wpm or 12 cps."
    durations = (df.ms_end - df.ms_start) / 1000
    lengths = df.text.str.len()
    too_fast = durations <= lengths / 24
    too_slow = durations > lengths / 6 + 10
    df.drop(df[too_fast | too_slow].index, inplace=True)


def remove_invalid_phrases(df: pd.DataFrame):
    df.drop(df[df.text.str.contains("CPossible")].index, inplace=True)


def filter_foreign_languages(df: pd.DataFrame, minimum_consecutive=4,
                             accepted_languages=("no", "nb", "no", "da", "sv")):
    """
    English occurs particularly often, mostly due to song lyrics.
    This method filters out foreign languages from the dataframe, but only when enough occur in a row.
    """
    # Most likely language is English
    is_english = df.languages.str.match(f"({'|'.join(accepted_languages)}),")

    # Produces the number of consecutive English transcriptions in the current "group"
    consecutive = is_english.groupby((is_english != is_english.shift()).cumsum()).transform("sum") * is_english

    # To avoid dropping misclassified lines, only drop when there are many consecutive English predictions in a row
    df.drop(df[consecutive >= minimum_consecutive].index, inplace=True)


def ms_to_stamp(ms):
    hour = ms // (60 * 60 * 1000)
    ms = ms % (60 * 60 * 1000)
    min = ms // (60 * 1000)
    ms = ms % (60 * 1000)
    sec = ms // 1000
    ms = ms % 1000
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(hour, min, sec, ms)


def to_vtts(df, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    it = tqdm(df.groupby("program"), desc=os.path.basename(output_folder))
    for program, sub_df in it:
        out_path = os.path.join(output_folder, f"{program}.vtt")
        if os.path.isfile(out_path):
            continue
        with open(out_path, "w") as out_file:
            n = 1
            out_file.write("WEBVTT\n")
            for _, row in sub_df.iterrows():
                timestamp = f"{ms_to_stamp(row.ms_start)} --> {ms_to_stamp(row.ms_end)}"
                text = row.text.replace("|", "\n")

                out_file.write(f"\n{n}\n{timestamp}\n{text}\n")
                n += 1
        it.set_postfix_str(program)


def main():
    # TODO make arguments

    # print("Making dataframe...")
    # make_dataframe("out/vtts/")

    print("Loading file...")
    df = pd.read_csv("../lines.tsv", sep="\t", quoting=3)
    df.text = df.text.fillna("").str.strip()

    df_clean = df.copy()
    # print("Removing invalid durations...")
    # remove_invalid_durations(df_clean)

    print("Separating ttv/nor...")
    ttv, nor = separate_nor_ttv(df_clean)

    to_vtts(ttv, "../out/vtts_transcribe")
    to_vtts(nor, "../out/vtts_translate")

    exit(0)

    for name, df_clean in ("nor", nor.copy()), ("ttv", ttv.copy()):
        print("Removing italics...")
        remove_italics(df_clean)

        print("Removing inaudible...")
        remove_inaudible(df_clean)

        print("Removing many lines...")
        remove_many_lines(df_clean)

        for overlapping in (False, True):
            df_clean2 = df_clean.copy()
            print("Removing splits...")
            remove_splits(df_clean2, drop_overlapping=not overlapping)
            overlapping = "+overlap" * overlapping
            df_clean2.to_csv(f"{name}{overlapping}.tsv", sep="\t", index=False, quoting=3)

            print("Predicting languages...")
            add_languages(df_clean2)
            df_clean2.to_csv(f"{name}+lang{overlapping}.tsv", sep="\t", index=False, quoting=3)

            print("Filtering foreign languages...")
            filter_foreign_languages(df_clean2)
            df_clean2.to_csv(f"{name}+lang_filter{overlapping}.tsv", sep="\t", index=False, quoting=3)

    print("Done!")


if __name__ == '__main__':
    main()
