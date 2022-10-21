# Class for Processing VTT files
The class contains the following methods:

separate_nor_ttv(df: pd.DataFrame):
    """
    Typically TTV is a superset of NOR, so anything in NOR is a spoken foreign language translated to Norwegian text,
     while TTV is Norwegian OR foreign language to Norwegian text.
    This method subtracts any duplicates from TTV, leaving only the Norwegian->Norwegian parts,
     and returns two separate dataframes.
    Note that NOR will sometimes include Norwegian->Sami as well
    """

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

def remove_inaudible(df: pd.DataFrame):
    """
    Some special cases of patterns in subtitles that are not spoken out loud.
    """

def remove_splits(df: pd.DataFrame, drop_overlapping=False):
    """
    This method goes through and concatenates any lines that belong together,
     e.g. sentences over two lines, sentences over several consecutive lines in different timestamps
     (continuation is denoted by ending a line and starting the next line with -).
    If two lines have dashes that denotes multiple speakers speaking simultaneously or in rapid succession,
     these can optionally be filtered out using `drop_overlapping`
    (note: generation of dataframe replaces newlines with a pipe for parsing)
    """

def add_languages(df: pd.DataFrame):
    """
    Add column with language predictions.
    Makes a comma separated string of language_code,probability for up to 3 languages
    """

def remove_invalid_durations(df: pd.DataFrame):
    """
    Some durations will clearly not make sense, either by being too short (some are even negative) or being too long.
    This filters out some of the more extreme durations based on text length.
    """

def filter_foreign_languages(df: pd.DataFrame, minimum_consecutive=4,
                             accepted_languages=("no", "nb", "no", "da", "sv")):
    """
    English occurs particularly often, mostly due to song lyrics.
    This method filters out foreign languages from the dataframe, but only when enough occur in a row.
    """


