from typing import List

from musif.config import ExtractConfiguration
from musif.extract.constants import DATA_SCORE
from .constants import *


def update_score_objects(
    score_data: dict,
    parts_data: List[dict],
    cfg: ExtractConfiguration,
    parts_features: List[dict],
    score_features: dict,
):

    score = score_data[DATA_SCORE]
    composer = score.metadata.composer
    score_features.update({COMPOSER: composer.title()})


def update_part_objects(
    score_data: dict, part_data: dict, cfg: ExtractConfiguration, part_features: dict
):
    pass
