from typing import List

from music21.text import TextBox

from musif.config import Configuration
from musif.extract.constants import DATA_SCORE

from .constants import *


def update_score_objects(
    score_data: dict,
    parts_data: List[dict],
    cfg: Configuration,
    parts_features: List[dict],
    score_features: dict,
):
    """
    get variables from file_name
    returns a dictionary so it can be easily input in a df
    """

    score = score_data[DATA_SCORE]
    librettist = None
    for textbox in score.getElementsByClass(TextBox):
        lines = textbox.content.split("\n")
        for line in lines:
            if "Text:" in line:
                librettist = _extract_name(line.split(":")[1])
    score_features.update(
        {
            LIBRETTIST: librettist,
        }
    )


def update_part_objects(
    score_data: dict, part_data: dict, cfg: Configuration, part_features: dict
):
    pass


def _extract_name(text: str) -> str:
    chars = [char for char in text if char.isalpha() or char == " "]
    name = "".join(chars).strip()
    return name
