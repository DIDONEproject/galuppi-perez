from math import floor
from os import path

import musif.extract.constants as C
from music21.stream import Measure, Score

from ..utils import get_ariaid


def execute(cfg, data):
    score: Score = data[C.DATA_SCORE]

    # extracting theme_a information from metadata
    aria_id = get_ariaid(path.basename(data[C.DATA_FILE]))
    last_measure = 1000000
    for d in cfg.scores_metadata[cfg.theme_a_metadata]:
        if d["Id"] == aria_id:
            last_measure = floor(float(d.get(cfg.end_of_theme_a, last_measure)))
            break

    # removing everything after end of theme A
    for part in score.parts:
        read_measures = 0
        elements_to_remove = []
        for measure in part.getElementsByClass(Measure):  # type: ignore
            read_measures += 1
            if read_measures > last_measure:
                elements_to_remove.append(measure)
        part.remove(targetOrList=elements_to_remove)  # type: ignore
    if cfg.is_requested_musescore_file() and data[C.DATA_MUSESCORE_SCORE] is not None:
        data[C.DATA_MUSESCORE_SCORE] = data[C.DATA_MUSESCORE_SCORE].loc[
            data[C.DATA_MUSESCORE_SCORE]["mn"] <= last_measure
        ]
        data[C.DATA_MUSESCORE_SCORE].reset_index(inplace=True, drop=True)
    return data
