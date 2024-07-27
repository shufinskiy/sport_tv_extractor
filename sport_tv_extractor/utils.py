from pathlib import Path
from typing import Union, Optional
import time
from functools import wraps

import numpy as np
import pandas as pd
from scipy.special import softmax
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def to_real_time(pred_shape: int, second_step: float, fps: int) -> np.ndarray:
    if second_step >= 1:
        plus_one = 0 if second_step != 4 else 1
        if second_step == 1:
            screen_frame = 24
        else:
            screen_frame = 49
        return np.array(np.arange(stop=pred_shape * second_step, step=second_step) + (screen_frame / fps)) + plus_one
    else:
        frame_step = np.floor(np.floor(fps * second_step))
        if second_step == 0.5:
            mask = np.array([0, 0])
        else:
            mask = np.array([0, 0, 1, 1])
        v1 = np.tile(np.arange(np.floor(frame_step / 2), fps, frame_step) + mask, int(pred_shape * second_step)) / fps
        v2 = np.repeat(np.arange(0, int(pred_shape * second_step)), int(1/second_step))
        return v1 + v2


class CustomImageFolder(Dataset):
    """A class for loading images

    This class loads images obtained from a video file after the method is working FFMpeg.cut_frames

    Attributes:
        paths List: List of paths to images
        transform Optional[transforms]: Transformations for images, if they are needed (default None)
    """

    def __init__(self,
                 path: Union[str, Path],
                 transform: Optional[transforms.Compose] = None):
        self.paths = sorted(list(Path(path).iterdir()), key=lambda x: int(x.stem.split('-')[1]))
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
    """A class for creating a data frame with classification results.

    A class creates a table based on image classification results,
    calculates the beginning, end, and duration of fragments from
    the main and other cameras, and leaves only data about fragments
    from the main camera.

    Attributes:
        prediction np.ndarray: array of class probabilities
    """

    def __init__(self, prediction: np.ndarray):
        self.prediction = prediction
        self.df = (
            pd.DataFrame(softmax(self.prediction, axis=1), columns=['mark_0', 'mark_1'])
        )

    def img_classification_df(self, second_step: float, fps: int) -> None:
        """Adding time and lag/lead mark information in dataframe

        Args:
            fps int: frame per second

        Returns:
            None
        """
        self.df = (
            self.df
            .assign(mark=self.prediction.argmax(axis=1))
            .assign(real_time=to_real_time(self.prediction.shape[0], second_step, fps))
            .assign(
                prev_value=lambda df_: df_.mark.shift(1, fill_value=1),
                next_value=lambda df_: df_.mark.shift(-1, fill_value=1)
            )
        )

    def main_camera_parts(self, skip_time: int) -> None:
        """Creating a table with data about fragments from the main camera

        The method finds continuous chains of images from the main camera
        and combines them into fragments, adding information about their
        beginning and ending. Information about fragments from other cameras
        is deleted. It remains if the fragment from the other camera is
        less than skip_time.

        Args:
            skip_time int: Time in seconds. For a fragment from other cameras
            to be preserved, it must be less than skip_time

        Returns:
            None
        """
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
            .assign(
                ind=lambda df_: [idx if diff > skip_time else None for diff, idx in zip(
                    df_.diff_time.shift(-1, fill_value=1000),
                    df_.index
                )]
            )
            .assign(
                ind=lambda df_: df_.ind.bfill()
            )
            .groupby(['ind'], as_index=False).agg(
                start_time=('start_time', 'min'),
                start_index=('start_index', 'min'),
                end_time=('end_time', 'max'),
                end_index=('end_index', 'max'),
            )
            .assign(
                duration=lambda df_: df_.end_time - df_.start_time,
                diff_time=lambda df_: df_.start_time - df_.end_time.shift(1, fill_value=-100)
            )
            .drop(columns=['ind', 'diff_time'])
            .reset_index(drop=True)
        )

    def upd_main_camera(self,
                        new_start_time: np.ndarray,
                        new_end_time: np.ndarray) -> None:
        """Update timestamps start and end video from main camera

        Args:
            new_start_time np.ndarray: new timestamps of the beginning of fragments from the main camera
            new_end_time np.ndarray: new timestamps of the ending of fragments from the main camera

        Returns:
            None
        """
        self.df = (
            self.df
            .assign(
                start_time=new_start_time,
                end_time=new_end_time,
                duration=lambda df_: df_.end_time - df_.start_time
            )
            .loc[:, ['start_time', 'start_index', 'end_time', 'end_index', 'duration']]
        )


def time_complete(text):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, 'logging', False):
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                print(f"{text} {round(end_time - start_time, 3)}")
            else:
                result = func(self, *args, **kwargs)
            return result
        return wrapper
    return decorator
