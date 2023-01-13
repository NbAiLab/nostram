import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from pandarallel import pandarallel

pandarallel.initialize(use_memory_fs=False)


def _find_abs(a, b):
    if isinstance(a, str) and isinstance(b, str):
        a, b = text_to_counter(a), text_to_counter(b)
    intersect = a & b
    abs_a = sum(a.values()) if isinstance(a, Counter) else len(a)
    abs_b = sum(b.values()) if isinstance(b, Counter) else len(b)
    abs_i = sum(intersect.values()) if isinstance(intersect, Counter) else len(intersect)
    abs_ab = sum(a[k] * b[k] for k in intersect)
    return abs_a, abs_b, abs_i, abs_ab


def text_to_counter(text, multiset=True):
    init = Counter if multiset else set
    clean = re.sub(r'[^\w\s]', '', text).lower()
    return init(clean.split())


def match_percentage(source, target):
    source_set = text_to_counter(source)
    target_set = text_to_counter(target)
    common = source_set & target_set
    return len(common) / len(source_set)


def cosine_similarity(a, b):
    abs_a, abs_b, abs_i, abs_ab = _find_abs(a, b)
    if 0 == abs_a == abs_b:
        return 1.
    return abs_ab / (math.sqrt(abs_a * abs_b))


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def relative_levenshtein(a, b):
    return levenshtein_distance(a, b) / max(len(a), len(b))


def jaro_winkler_distance(st1, st2):
    """
    Compute Jaro-Winkler distance between two strings.
    https://rosettacode.org/wiki/Jaro-Winkler_distance#Python
    """
    if len(st1) < len(st2):
        st1, st2 = st2, st1
    len1, len2 = len(st1), len(st2)
    if len2 == 0:
        return 0.0
    delta = max(0, len2 // 2 - 1)
    flag = [False for _ in range(len2)]  # flags for possible transpositions
    ch1_match = []
    for idx1, ch1 in enumerate(st1):
        for idx2, ch2 in enumerate(st2):
            if idx1 + delta >= idx2 >= idx1 - delta and ch1 == ch2 and not flag[idx2]:
                flag[idx2] = True
                ch1_match.append(ch1)
                break

    matches = len(ch1_match)
    if matches == 0:
        return 1.0
    transpositions, idx1 = 0, 0
    for idx2, ch2 in enumerate(st2):
        if flag[idx2]:
            transpositions += (ch2 != ch1_match[idx1])
            idx1 += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0
    commonprefix = 0
    for i in range(min(4, len2)):
        commonprefix += (st1[i] == st2[i])

    return 1.0 - (jaro + commonprefix * 0.1 * (1 - jaro))


def main():
    df = pd.read_json("../nrk.json", lines=True, nrows=100_000)
    transcriptions = pd.read_json("../wav2vec_transcripts.jsonl", lines=True, nrows=100_000)

    df = df.merge(transcriptions, left_on="id", right_on="file", suffixes=("", "_transcription"))

    mat_per = df.apply(lambda row: match_percentage(row["text"], row["text_transcription"]), axis=1)
    cos_sim = df.apply(lambda row: cosine_similarity(row["text"], row["text_transcription"]), axis=1)
    rel_lev = df.apply(lambda row: 1 - relative_levenshtein(row["text"], row["text_transcription"]), axis=1)
    jar_win = df.apply(lambda row: 1 - jaro_winkler_distance(row["text"], row["text_transcription"]), axis=1)
    breakpoint()


if __name__ == '__main__':
    main()
