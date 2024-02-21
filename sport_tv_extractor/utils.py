from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from PIL import Image
from torch.utils.data import Dataset


class CustomImageFolder(Dataset):

    def __init__(self, paths, transform=None):
        self.paths = sorted(list(Path(paths).iterdir()), key=lambda x: int(x.stem.split('-')[1]))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


class ExtractorDF(object):

    def __init__(self, prediction, paths):
        self.prediction = prediction
        self.paths = paths
        self.df = (
            pd.DataFrame(softmax(self.prediction, axis=1), columns=['mark_0', 'mark_1'])
            .assign(path=self.paths)
        )

    def img_classification_df(self, fps: int) -> None:
        self.df = (
            self.df
            .loc[:, ['path', 'mark_0', 'mark_1']]
            .assign(mark=self.prediction.argmax(axis=1))
            .assign(sec=lambda df_: [int(str(x).split('-')[1].rstrip('.jpeg')) for x in df_.path])
            .assign(real_time=lambda df_: [x - 1 + 24 / fps for x in df_.sec])
            .assign(
                prev_value=lambda df_: df_.mark.shift(1, fill_value=1),
                next_value=lambda df_: df_.mark.shift(-1, fill_value=1)
            )
        )

    def main_camera_parts(self, skip_time: int) -> None:
        df = (
            self.df
            .reset_index()
            .pipe(lambda df_: df_.loc[(df_.mark == 0) & ((df_.prev_value == 1) | (df_.next_value == 1))])
            .pipe(lambda df_: df_.loc[(df_.prev_value == 0) | (df_.next_value == 0)])
            .reset_index(drop=True)
        )

        start = (
            df
            .pipe(lambda df_: df_.loc[df_['prev_value'] == 1, ['real_time', 'index']])
            .reset_index(drop=True)
            .rename(
                columns={
                    'index': 'start_index',
                    'real_time': 'start_time'
                }
            )
        )
        end = (
            df
            .pipe(lambda df_: df_.loc[df_['next_value'] == 1, ['real_time', 'index']])
            .reset_index(drop=True)
            .rename(
                columns={
                    'index': 'end_index',
                    'real_time': 'end_time'
                }
            )
        )

        self.df = (
            pd.concat([start, end], axis=1)
            .assign(
                duration=lambda df_: df_.end_time - df_.start_time,
                diff_time=lambda df_: df_.start_time - df_.end_time.shift(1, fill_value=-100)
            )
            .assign(start_time=lambda df_: [x if z <= skip_time else y for x, y, z in zip(df_.start_time.shift(1),
                                                                                          df_.start_time,
                                                                                          df_.diff_time)])
            .drop_duplicates(subset=['start_time'], keep='last')
            .assign(
                start_index=lambda df_: [np.floor(x).astype('int') for x in df_.start_time],
                duration=lambda df_: df_.end_time - df_.start_time
            )
            .drop(columns='diff_time')
            .reset_index(drop=True)
        )

    def upd_main_camera(self, new_start_time, new_end_time) -> None:
        self.df = (
            self.df
            .assign(
                start_time=new_start_time,
                end_time=new_end_time,
                duration=lambda df_: df_.end_time - df_.start_time
            )
            .loc[:, ['start_time', 'start_index', 'end_time', 'end_index', 'duration']]
        )
