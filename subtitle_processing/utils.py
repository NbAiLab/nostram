from typing import Optional, List, Set, Union

import fasttext
from datasets.utils.download_manager import DownloadManager


NORDIC_LID_URL = "https://huggingface.co/NbAiLab/nb-nordic-lid/resolve/main/"
model_name = "nb-nordic-lid.bin"

model = fasttext.load_model(DownloadManager().download(NORDIC_LID_URL + model_name))
model_labels = set(label[-3:] for label in model.get_labels())


def detect_lang(
    text: str,
    langs: Optional[Union[List, Set]]=None,
    threshold: float=-1.0,
    return_proba: bool=False
) -> Union[str, Tuple[str, float]]:
    """
    This function takes in a text string and optional arguments for a list or
    set of languages to detect, a threshold for minimum probability of language
    detection, and a boolean for returning the probability of detected language.
    It uses a pre-defined model to predict the language of the text and returns
    the detected ISO-639-3 language code as a string. If the return_proba
    argument is set to True, it will also return a tuple with the language code
    and the probability of detection. If no language is detected, it will
    return "und" as the language code.

    Args:
    - text (str): The text to detect the language of.
    - langs (List or Set, optional): The list or set of languages to detect in 
        the text. Defaults to all languages in the model's labels.
    - threshold (float, optional): The minimum probability for a language to be
        considered detected. Defaults to `-1.0`.
    - return_proba (bool, optional): Whether to return the language code and
        probability of detection as a tuple. Defaults to `False`.

    Returns:
    str or Tuple[str, float]: The detected language code as a string, or a
        tuple with the language code and probability of detection if
        return_proba is set to True.
    """
    if langs:
        langs = set(langs)
    else:
        langs = model_labels
    raw_prediction = model.predict(text, threshold=threshold, k=-1)
    predictions = [
        (label[-3:], min(probability, 1.0))
        for label, probability in zip(*raw_prediction)
        if label[-3:] in langs
    ]
    if not predictions:
        return ("und", 1.0) if return_proba else "und"
    else:
        return predictions[0] if return_proba else predictions[0][0]

