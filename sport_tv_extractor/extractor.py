from pathlib import Path
import shutil
from typing import List, Optional, Dict, Any, Union, Callable, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models, transforms

from ffmpeg import FFMpeg
from utils import CustomImageFolder, ExtractorDF

TMP_DIR = '/home/shuf91/env/video_sport_game/package/left_right_check/'
URL_RESNET = 'https://github.com/shufinskiy/sport_extractor_models/raw/main/models/main_camera_extractor.pt'


class ExtractorBroadcast(object):
    """
    Attributes:
        path
        output_name
        skip_time
        device
        img_dir
        video_dir
        model_dir
        high_accuracy
        rm_files
        batch_size
        ffmpeg_v
        ffmpeg
        model
        transformation
        prediction
    """

    def __init__(self,
                 path: str,
                 output_name: str,
                 skip_time: int = 1,
                 device: str = 'cpu',
                 img_dir: str = 'images',
                 video_dir: str = 'video',
                 model_dir: str = 'models',
                 high_accuracy: bool = True,
                 rm_tmp_files: bool = True,
                 batch_size: int = 128,
                 ffmpeg_verbose: str = '-loglevel quiet -stats',
                 model: Optional[models.ResNet] = None,
                 transformation: Optional[transforms.Compose] = None,
                 prediction: Optional[np.ndarray] = None):
        self.path = path
        self.output_name = output_name
        self.skip_time = skip_time
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device != 'cpu' else device
        self.img_dir = img_dir
        self.video_dir = video_dir
        self.model_dir = model_dir
        self.high_accuracy = high_accuracy
        self.rm_files = rm_tmp_files if isinstance(rm_tmp_files, List) else [rm_tmp_files] * 2
        self.batch_size = batch_size
        self.ffmpeg_v = ffmpeg_verbose
        self.ffmpeg = FFMpeg(
            path=self.path,
            img_dir=self.img_dir,
            video_dir=self.video_dir,
            output_name=self.output_name,
            device=self.device,
            verbose_mode=self.ffmpeg_v,
            rm_tmp_image=self.rm_files[0],
            rm_tmp_video=self.rm_files[1]
        )
        self.model = model
        self.transformation = transformation
        self.prediction = prediction

    def main_camera_video(self) -> None:
        """

        Returns:

        """

        if self.model is None:
            self.model = self.init_model()

        if self.transformation is None:
            self.transformation = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if self.prediction is None:
            self.ffmpeg.cut_frames()
            self.prediction = self.classification_images(self.model, self.transformation)

        self.ffmpeg.get_fps()
        self.ffmpeg.get_num_frame()

        data = self.create_time_information(self.prediction)

        if self.high_accuracy:
            self.time_high_accuracy(data, self.model, self.transformation)

        self.ffmpeg_cut_videos(data)

        self.ffmpeg.concat_videos()

        self.ffmpeg.rm_tmp_files()

    def download_model_dict(self, url: str = URL_RESNET, progress: bool = False) -> Dict[str, Any]:
        """

        Args:
            url:
            progress:

        Returns:

        """
        return torch.hub.load_state_dict_from_url(url=url, model_dir=self.model_dir, progress=progress)

    def init_model(self) -> models.ResNet:
        """

        Returns:

        """
        model = models.resnet18()
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(self.download_model_dict())
        model.eval()
        model.to(self.device)

        return model

    def prediction_image(self,
                         for_pred: np.ndarray,
                         model: models.ResNet,
                         transformation: transforms.Compose,
                         tmp_dir: str = TMP_DIR) -> np.ndarray:
        """

        Args:
            for_pred:
            model:
            transformation:
            tmp_dir:

        Returns:

        """
        path = Path(tmp_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.ffmpeg.loop_cut_frames(path, for_pred)

        dataset = CustomImageFolder(path, transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = self.class_prediction(model, dataloader, shape=len(dataset))
        shutil.rmtree(path)
        return prediction

    def step_frames(self,
                    left_frames: np.ndarray,
                    right_frames: np.ndarray,
                    model: models.ResNet,
                    transformation: transforms.Compose,
                    tmp_dir: str = TMP_DIR) -> Union[np.ndarray, Callable]:
        """

        Args:
            left_frames:
            right_frames:
            model:
            transformation:
            tmp_dir:

        Returns:

        """

        if (left_frames.shape[1] == 1) and (right_frames.shape[1] == 1):
            for_pred_left = left_frames.reshape(-1, ) / self.ffmpeg.fps
            for_pred_right = right_frames.reshape(-1, ) / self.ffmpeg.fps
            for_pred = np.sort(np.concatenate((for_pred_left, for_pred_right)))

            prediction = self.prediction_image(for_pred, model, transformation, tmp_dir)

            left_pred = prediction[::2]
            right_pred = prediction[1::2]

            left_frames = np.array(
                [t if left_pred[i] == 0 else t + 1 for i, t in enumerate(left_frames.reshape(-1, ))])

            right_frames = np.array(
                [t if right_pred[i] == 0 else t - 1 for i, t in enumerate(right_frames.reshape(-1, ))])
            return np.sort(np.concatenate((left_frames, right_frames))) / self.ffmpeg.fps

        m = left_frames.shape[1] // 2
        low = m
        high = m + 1 if left_frames.shape[1] % 2 == 1 else m

        for_pred_left = left_frames[:, m] / self.ffmpeg.fps
        for_pred_right = right_frames[:, m] / self.ffmpeg.fps
        for_pred = np.sort(np.concatenate((for_pred_left, for_pred_right)))

        prediction = self.prediction_image(for_pred, model, transformation, tmp_dir)
        left_pred = prediction[::2]
        right_pred = prediction[1::2]

        left_frames = np.vstack([t[:low] if left_pred[i] == 0 else t[high:]
                                 for i, t in enumerate(left_frames)])

        right_frames = np.vstack([t[:low] if right_pred[i] == 1 else t[high:]
                                  for i, t in enumerate(right_frames)])

        return self.step_frames(left_frames, right_frames, model, transformation, tmp_dir)

    def class_prediction(self,
                         model: models.ResNet,
                         dataloader: torch.utils.data.DataLoader,
                         shape: Union[int, Tuple[int, int]]) -> np.ndarray:
        """

        Args:
            model:
            dataloader:
            shape:

        Returns:

        """
        prediction = np.empty(shape)
        for i, inputs in enumerate(dataloader):
            with torch.set_grad_enabled(False):
                inputs = inputs.to(self.device)
                preds = model(inputs).cpu()
                preds = preds.argmax(dim=1) if isinstance(shape, int) else preds.numpy()
                prediction[i * self.batch_size:(i + 1) * self.batch_size] = preds

        return prediction

    def classification_images(self,
                              model: models.ResNet,
                              transformation: transforms.Compose) -> np.ndarray:
        """

        Args:
            model:
            transformation:

        Returns:

        """
        dataset = CustomImageFolder(self.img_dir, transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = self.class_prediction(model, dataloader, shape=(len(dataset), 2))

        return prediction

    def create_time_information(self, prediction: np.ndarray) -> ExtractorDF:
        """

        Args:
            prediction:

        Returns:

        """
        data = ExtractorDF(prediction)
        data.img_classification_df(self.ffmpeg.fps)
        data.main_camera_parts(self.skip_time)

        return data

    def time_high_accuracy(self,
                           data: ExtractorDF,
                           model: models.ResNet,
                           transformation: transforms.Compose) -> ExtractorDF:
        """

        Args:
            data:
            model:
            transformation:

        Returns:

        """
        frames = np.array([24 + (self.ffmpeg.fps * i) for i in
                           range(np.floor(self.ffmpeg.num_frame / self.ffmpeg.fps).astype(np.int16))])

        st_fr = data.df.start_index.tolist()
        st_frames = np.array([frames[idx] for idx in st_fr])
        left_frames = np.clip(
            np.array([np.arange(st_frame - self.ffmpeg.fps + 1, st_frame) for st_frame in st_frames]),
            a_min=0,
            a_max=self.ffmpeg.num_frame - (100 / self.ffmpeg.fps)
        )

        end_fr = data.df.end_index.tolist()
        end_frames = np.array([frames[idx] for idx in end_fr])
        right_frames = np.clip(
            np.array([np.arange(st_frame + 1, st_frame + self.ffmpeg.fps) for st_frame in end_frames]),
            a_min=0,
            a_max=self.ffmpeg.num_frame - (100 / self.ffmpeg.fps)
        )

        new_time = self.step_frames(left_frames, right_frames, model, transformation)

        data.upd_main_camera(new_time[::2], new_time[1::2])

        return data

    def ffmpeg_cut_videos(self, data: ExtractorDF) -> None:
        """

        Args:
            data:

        Returns:

        """
        for i, row in enumerate(data.df.itertuples()):
            self.ffmpeg.cut_videos(row.start_time, row.duration, i)

        with open(f'{self.video_dir}/file.txt', 'w', encoding='utf-8') as file_desc:
            for i in range(data.df.shape[0]):
                file_desc.write(f"file 'video_{i}.mkv'\n")
