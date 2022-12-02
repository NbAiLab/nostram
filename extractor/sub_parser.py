"""
Parse subtitle file (srt, vtt) and convert to JSON

NORCE Research Institute 2022, Njaal Borch <njbo@norceresearch.no>
Licensed under GPL v3
"""

import re
import os.path


class SubParser:
    """Parse VTT/SRT files."""

    def __init__(self):
        self.items = []

    @staticmethod
    def time2sec(t):
        ms = t.split(",")[1]
        t = t[:-len(ms) - 1]
        h, m, s = t.split(":")
        ts = int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.)
        return ts

    def load_srt(self, filename, default_who=None):
        if os.path.exists(filename) == False:
            return None

        with open(filename, "rb") as f:

            start = end = None
            text = ""

            lines = [line.decode("utf-8").strip().replace("—", "-") for line in f.readlines()]
            for line in lines:
                if re.match(r"^\d+$", line) and not (start or end or text):
                    # Just the index, ignore
                    continue

                m = re.match(r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)", line.replace(".", ","))
                if m:
                    # if start and not end:
                    #     _, end = m.groups()
                    #     text += "<p>"
                    # else:
                    start, end = m.groups()
                    text = ""
                    continue

                if start and end and line:
                    if text:
                        text += "<br>" + line
                    else:
                        text += line

                if start and end and (not line or line == lines[-1]):
                    # if line.replace("</i>", "").endswith("-"):
                    #     end = None
                    #     continue
                    # print("End of comment", text)
                    # End of comment
                    s = {
                        "start": SubParser.time2sec(start),  # Used to be a +0.01 sometimes, not sure why
                        "end": SubParser.time2sec(end),
                        "text": text
                    }
                    if default_who:
                        s["who"] = default_who

                    self.items.append(s)
                    text = ""
                    start = end = None

        return self.items


if __name__ == "__main__":
    import sys

    parser = SubParser()
    parser.load_srt(sys.argv[1])
    print(parser.items)
