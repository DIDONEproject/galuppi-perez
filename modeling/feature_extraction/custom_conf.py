from glob import glob
from os import path

from musif.common._utils import read_dicts_from_csv
from musif.config import ExtractConfiguration


class CustomConf(ExtractConfiguration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_metadata()

    def _load_metadata(self) -> None:
        self.scores_metadata = {
            path.basename(file): read_dicts_from_csv(file)
            for file in glob(path.join(self.metadata_dir, "score", "*.csv"))
        }
        if not self.scores_metadata:
            print(
                "\nMetadata could not be loaded properly!! Check metadata path in config file.\n"
            )
        self.characters_gender = read_dicts_from_csv(
            path.join(self.internal_data_dir, "characters_gender.csv")
        )

        # nnot used anymore or used in report only
        # self.abbreviation_to_sound = {
        #     abbreviation: sound
        #     for sound, abbreviation in self.sound_to_abbreviation.items()
        # }
        # self.translations_cache = read_object_from_json_file(
        #     path.join(self.internal_data_dir, "translations.json")
        # )
        # self.sorting_lists = read_object_from_json_file(
        #     path.join(self.internal_data_dir, "sorting_lists.json")
        # )
