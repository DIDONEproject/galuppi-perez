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
