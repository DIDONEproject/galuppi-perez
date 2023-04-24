from typing import List
from musif.extract.constants import ID

from musif.config import ExtractConfiguration
from .constants import CHARACTER
from musif.logs import lwarn


def update_score_objects(
    score_data: dict,
    parts_data: List[dict],
    cfg: ExtractConfiguration,
    parts_features: List[dict],
    score_features: dict,
):
    features = {}
    for group_name, group_features in cfg.scores_metadata.items():
        existing_metadata_id = score_features.get(ID)
        if not existing_metadata_id:
            continue
        if len(group_features) == 0:
            continue
        if ID not in group_features[0]:
            continue
        for item_features in group_features:
            if item_features[ID] != existing_metadata_id:
                continue
            for key in item_features:
                if (key in score_features) and (key != ID):
                    lwarn(
                        f"Column {key} exists both in metadata and in existing features"
                    )
                    continue
                features[key] = item_features[key]

    extract_character(score_data, parts_data, features)
    return score_features.update(features)


# TODO: Label_passion 'Label_Passions' and 'Label_Sentiment' assigned here?


def extract_character(score_data, parts_data, features):
    character = []
    for part in parts_data:
        if part["family"] == "voice":
            name = part["part"].partName.capitalize()
            character.append(name)

    features[CHARACTER] = "&".join(character)


def update_part_objects(
    score_data: dict, part_data: dict, cfg: ExtractConfiguration, part_features: dict
):
    pass
