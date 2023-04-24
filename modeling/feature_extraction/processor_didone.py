import difflib
import os
from typing import Union

import numpy as np
import pandas as pd
from musif.common._utils import read_dicts_from_csv
from musif.common.sort import sort_columns
from musif.extract.basic_modules.scoring.constants import (
    INSTRUMENTATION,
    ROLE_TYPE,
    SCORING,
    VOICES,
)
from musif.extract.constants import ID, WINDOW_ID
from musif.extract.features.ambitus.constants import (
    HIGHEST_NOTE_INDEX,
    LOWEST_NOTE_INDEX,
)
from musif.extract.features.core.constants import FILE_NAME
from musif.extract.features.harmony.constants import HARMONY_AVAILABLE
from musif.extract.features.key.constants import KEY, KEY_SIGNATURE, KEY_SIGNATURE_TYPE
from musif.extract.features.prefix import get_part_prefix, get_sound_prefix
from musif.extract.features.texture.constants import TEXTURE
from musif.logs import perr, pinfo, pwarn
from musif.process.constants import voices_list_prefixes
from musif.process.processor import DataProcessor
from pandas import DataFrame
from tqdm import tqdm

from .custom_basic_modules.composer.handler import COMPOSER
from .custom_basic_modules.file_name.constants import *
from .custom_basic_modules.file_name.constants import ARIA_ID, ARIA_LABEL
from .custom_basic_modules.metadata.constants import *

"""Dictionary to assing label prefix to columns that need to bein the _labels.csv file
to run analysis."""
label_by_col = {
    "Basic_passion": "Label_BasicPassion",
    "PassionA": "Label_PassionA",
    "PassionB": "Label_PassionB",
    "Value": "Label_Value",
    "Value2": "Label_Value2",
    "Time": "Label_Time",
}

"""Columns to be placed at the beginning of the exported DataFrame"""
priority_columns = [
    FILE_NAME,
    ARIA_OPERA,
    ARIA_LABEL,
    ARIA_NAME,
    ARIA_ACT,
    ARIA_SCENE,
    ACTANDSCENE,
    ARIA_YEAR,
    ARIA_DECADE,
    COMPOSER,
    ARIA_CITY,
    TERRITORY,
    CHARACTER,
    GENDER,
    FORM,
    KEY,
    KEY_SIGNATURE,
    KEY_SIGNATURE_TYPE,
    INSTRUMENTATION,
    SCORING,
    VOICES,
]

"""Columns that will be included in the _metadata.csv exported file"""
metadata_columns = [
    FILE_NAME,
    ARIA_OPERA,
    ARIA_LABEL,
    ARIA_NAME,
    ARIA_ACT,
    ARIA_SCENE,
    ACTANDSCENE,
    ARIA_YEAR,
    ARIA_DECADE,
    COMPOSER,
    ARIA_CITY,
    TERRITORY,
    CHARACTER,
    GENDER,
    HARMONY_AVAILABLE,
]


class DataProcessorDidone(DataProcessor):
    def __init__(self, info: Union[str, DataFrame], *args, **kwargs):
        super().__init__(info, *args, **kwargs)

    def process(self) -> DataFrame:
        if self._post_config.delete_failed_files:
            self.delete_previous_items()

        if self._post_config.merge_voices:
            self.merge_voices()

        super().process()

        # Delete Vn when it is alone
        to_delete = [i for i in self.data.columns if get_part_prefix("Vn") in i]
        to_delete.append(f"PartVnI__PartVoice__{TEXTURE}")
        self.data.drop(columns=to_delete, inplace=True, errors="ignore")

        pinfo("\nScanning info looking for missing data...")
        self._scan_dataframe()
        # self.save(self.destination_route)
        return self.data

    def assign_labels(self) -> None:
        """
        Crosses passions labels from Passions.csv file with the DataFrame so every row (aria)
        gets assigned to its own Label
        """
        passions = read_dicts_from_csv(
            os.path.join(self._post_config.internal_data_dir, "Passions.csv")
        )

        data_by_aria_label = {
            label_data["Label"]: label_data for label_data in passions
        }
        for col, label in label_by_col.items():
            values = []
            for _, row in self.data.iterrows():
                data_by_aria = data_by_aria_label.get(row[ARIA_LABEL])
                label_value = data_by_aria[col] if data_by_aria else None
                values.append(label_value)
            self.data[label] = values

        if self._post_config.split_passionA:
            split_passion_A(self.data)

    def preprocess_data(self) -> None:
        """
        Adds labels to arias. Cleans data and removes columns with no information or
        rows without assigned Label
        """
        self.assign_labels()
        if "Label_Passions" in self.data:
            del self.data["Label_Passions"]
        if "Label_Sentiment" in self.data:
            del self.data["Label_Sentiment"]

        print(
            "Deleted arias without passion: ",
            self.data[self.data["Label_BasicPassion"].isnull()].shape[0],
        )
        self.data = self.data[~self.data["Label_BasicPassion"].isnull()]

        self.data.dropna(axis=1, how="all", inplace=True)

    def merge_voices(self) -> None:
        """
        Finds multiple singers arias (duetos/trietos) and calculates mean, max or min
        between them.
        Unifies all voices columns into SoundVoice_ columns.
        Also collapses PartBsI and PartBsII in one column.
        """
        pinfo("\nScanning voice columns")
        df_voices = self.data[
            [
                col
                for col in self.data.columns
                if any(substring in col for substring in voices_list_prefixes)
            ]
        ]
        self.data[df_voices.columns] = self.data[df_voices.columns].replace(
            "NA", np.nan
        )

        merge_single_voices(self.data)
        self.data = merge_duetos_trios(self.data)

        columns_to_delete = [
            i
            for i in self.data.columns.values
            if any(voice in i for voice in voices_list_prefixes)
        ]
        self.data.drop(columns_to_delete, axis=1, inplace=True)

        self.data = _join_double_bass(self.data)

    def delete_previous_items(self) -> None:
        """
        Deletes items from 'errors.csv' file in case they were not extracted properly
        """
        pinfo("\nDeleting items with errors...")
        errors_file = r"./errors.csv"
        if os.path.exists(errors_file):
            errors = pd.read_csv(
                errors_file, low_memory=False, encoding_errors="replace", header=0
            )[FILE_NAME].tolist()
            for item in errors:
                index = self.data.index[self.data[FILE_NAME] == item + ".xml"]
                if not index.empty:
                    self.data.drop(index, axis=0, inplace=True)
                    pwarn("Item {0} from errors.csv was deleted.".format(item))
        else:
            perr(
                '\nA file called "errors.csv" must be created containing FileNames to be deleted.'
            )

    def _scan_dataframe(self):
        # self.composer_counter = []
        # self.novoices_counter = []
        self._scan_composers()
        self._scan_voices()

    def _scan_voices(self):
        to_delete = self.data[VOICES].isna()
        self.data = self.data[~to_delete]

    def _scan_composers(self):
        composers_path = os.path.join(
            self._post_config.internal_data_dir, "composers.csv"
        )

        if os.path.exists(composers_path):
            composers = pd.read_csv(composers_path)
            composers = [i for i in composers.iloc[:, 0].to_list() if str(i) != "nan"]

            to_delete = []
            index = self.data.index
            if index.nlevels > 1:
                index = index.levels[0]
            for i, _ in enumerate(index):
                comp = self.data[COMPOSER][i]
                if not isinstance(comp, (pd.DataFrame, pd.Series)):
                    comp = pd.Series([comp])
                if pd.isnull(comp).any():
                    # self.composer_counter.append(self.data[FILE_NAME][i])
                    to_delete.append(i)
                elif comp.str.strip().isin(composers).any():
                    aria_name = self.data.loc[i, FILE_NAME]
                    corrections = comp.apply(
                        lambda x: self._get_close_matches(x, composers)
                    )
                    if corrections[0] == "NA":
                        # self.composer_counter.append(self.data[FILE_NAME][i])
                        to_delete.append(i)
                    elif any(corrections != comp):
                        pwarn(
                            f"Composer {comp[0]} in aria {aria_name[0]} was not found. Replaced with: {corrections[0]}"
                        )
                        self.data.loc[i, COMPOSER] = corrections
            self.data.drop(index=to_delete, inplace=True)
        else:
            perr("Composers file could not be found.")

    def _get_close_matches(self, comp, composers):
        correction = difflib.get_close_matches(comp, composers)
        correction = correction[0] if correction else "NA"
        return correction

    def save(self, dest_path, ext=".csv", ft="csv", **kwargs) -> None:

        super().save(dest_path, ft=ft, ext=ext)
        ft = "to_" + ft
        dest_path = str(dest_path)
        if ft == "to_csv":
            kwargs["index"] = False
        getattr(self.label_dataframe, ft)(dest_path + "_labels" + ext, **kwargs)
        getattr(self.metadata_dataframe, ft)(dest_path + "_metadata" + ext, **kwargs)
        getattr(self.features_dataframe, ft)(dest_path + "_features" + ext, **kwargs)
        getattr(self.data, ft)(dest_path + "_alldata" + ext, **kwargs)

    def _final_data_processing(self) -> None:
        super()._final_data_processing()
        self._split_metadata_and_labels()

    def _split_metadata_and_labels(self) -> None:
        self.data.rename(columns={ROLE_TYPE: "Label_" + ROLE_TYPE}, inplace=True)
        label_columns = list(self.data.filter(like="Label_", axis=1))
        self.label_dataframe = self.data[[ARIA_ID, WINDOW_ID] + label_columns]

        self.metadata_dataframe = self.data[[ARIA_ID, WINDOW_ID] + metadata_columns]
        self.data = sort_columns(self.data, [ARIA_ID, WINDOW_ID] + priority_columns)
        priority_columns.remove(KEY)
        priority_columns.remove(KEY_SIGNATURE)
        priority_columns.remove(KEY_SIGNATURE_TYPE)
        
        self.features_dataframe = self.data.drop(
            priority_columns + label_columns, axis=1, errors="ignore"
        )
        if not len(self.features_dataframe.columns) + len(priority_columns) + len(label_columns) == len(self.data.columns):
            perr('Mismatch found between column numbers of all different dataframes!')


def merge_single_voices(df: DataFrame) -> None:
    generic_sound_voice_prefix = get_sound_prefix("Voice")
    pinfo("\nJoining voice parts...")
    singer_columns = [
        i
        for i in df.columns.values
        if any(voice in i for voice in voices_list_prefixes)
    ]
    for col in singer_columns:
        singer_part = col.split("_")[0]
        generic_col = "_".join(col.split("_")[1:])
        formatted_col = col.replace(singer_part + "_", generic_sound_voice_prefix)
        if formatted_col in df:
            continue
        columns_to_merge = [
            i for i in singer_columns if "_".join(i.split("_")[1:]) == generic_col
        ]
        if all(df[columns_to_merge].dtypes == object):
            df[columns_to_merge] = df[columns_to_merge].astype(str)
            df[formatted_col] = df[columns_to_merge].sum(axis=1)
            df[formatted_col] = [i.replace("nan", "") for i in df[formatted_col]]
        else:
            for colum in columns_to_merge:
                df[colum].fillna(0, inplace=True)
            df[formatted_col] = df[columns_to_merge].sum(axis=1)


def _join_double_bass(df: DataFrame):
    df.drop([i for i in df.columns if "PartBsII" in i], axis=1, inplace=True)
    double_bass_columns = [i for i in df.columns if "PartBsI" in i]
    for col in double_bass_columns:
        formatted_col = col.replace("BsI_", "Bs_")
        df[formatted_col].fillna(0, inplace=True)
        if df[formatted_col].dtypes == object:
            df[formatted_col] = df[formatted_col].astype(str)
            df[col] = df[col].astype(str)
            df[formatted_col] = df[[col, formatted_col]].sum(axis=1)
            df[formatted_col] = [i.replace("nan", "") for i in df[formatted_col]]
            # df[formatted_col] = df[formatted_col].astype(float)
        else:
            df[col] = df[col].astype(float)
            df[formatted_col] = df[[formatted_col, col]].sum(axis=1)
    df.drop(double_bass_columns, axis=1, inplace=True)

    return df


def split_passion_A(data: DataFrame) -> None:
    data["Label_PassionA_primary"] = data["Label_PassionA"].str.split(";", expand=True)[
        0
    ]
    data["Label_PassionA_secundary"] = data["Label_PassionA"].str.split(
        ";", expand=True
    )[1]
    data["Label_PassionA_secundary"].fillna(
        data["Label_PassionA_primary"], inplace=True
    )
    data.drop("Label_PassionA", axis=1, inplace=True)


def merge_duetos_trios(df: DataFrame) -> None:
    generic_sound_voice_prefix = get_sound_prefix("Voice")

    df = df[df[VOICES].notna()]
    multiple_voices = df[df[VOICES].str.contains(",")]
    multiple_voices = _remove_repeated_voices(multiple_voices)
    pinfo(
        f"{multiple_voices.shape[0]} arias were found with duetos/trietos. Calculating averages."
    )
    voice_cols = [
        col
        for col in df.columns.values
        if any(voice in col for voice in voices_list_prefixes)
    ]

    for index in tqdm(multiple_voices.index):
        name = df.at[index, FILE_NAME]
        all_voices = df.at[index, VOICES].split(",")
        if all(x == all_voices[0] for x in all_voices):
            continue
        pinfo(f"\nMerging dueto/trieto {name}")
        first_voice = all_voices[0]
        columns_to_merge = [i for i in voice_cols if first_voice in i.lower()]
        for col in columns_to_merge:
            similar_cols = []
            formatted_col = col.replace(
                get_part_prefix(first_voice), generic_sound_voice_prefix
            )
            for j in range(0, len(all_voices)):
                similar_col = col.replace(
                    get_part_prefix(first_voice), get_part_prefix(all_voices[j])
                )
                if similar_col in df:
                    similar_cols.append(similar_col)
            if all(isinstance(x, str) for x in df.loc[index, similar_cols]):
                df.at[index, formatted_col] = df.loc[index, similar_cols][0]
            elif all(np.isnan(x) for x in df.loc[index, similar_cols]):
                df.drop(similar_cols, inplace=True, axis=1)
            elif HIGHEST_NOTE_INDEX in col or ("Largest" and "Asc") in col:
                df.at[index, formatted_col] = df.loc[index, similar_cols].max()
            elif LOWEST_NOTE_INDEX in col or ("Largest" and "Desc") in col:
                df.at[index, formatted_col] = df.loc[index, similar_cols].min()
            else:
                df.at[index, formatted_col] = df.loc[index, similar_cols].mean()
    return df


def _remove_repeated_voices(multiple_voices):
    repeated_voices_indexes = []
    for i, row in multiple_voices.iterrows():
        if all(x == row[VOICES].split(",")[0] for x in row[VOICES].split(",")):
            repeated_voices_indexes.append(i)
    multiple_voices = multiple_voices[
        ~multiple_voices.index.isin(repeated_voices_indexes)
    ]
    return multiple_voices
