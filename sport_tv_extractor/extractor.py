from pathlib import Path
import shutil
from typing import List, Optional, Dict, Any, Union, Callable, Tuple

import numpy as np
from scipy.special import softmax

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models, transforms

from ffmpeg import FFMpeg
from utils import CustomImageFolder, ExtractorDF, time_complete

TMP_DIR = '/home/shuf91/env/video_sport_game/package/left_right_check/'
URL_RESNET = 'https://github.com/shufinskiy/sport_extractor_models/raw/main/models/main_camera_extractor.pt'


class ExtractorBroadcast(object):
    """
    A class for creating filtered videos

    This class allows you to make a video from a raw video broadcast
    of a football match, in which non-game moments will be absent.

    Attributes:
        path str: Path to video
        output_name str: Name for output file
        skip_time int:
        second_step: float
        device str: Device on which must make video processing: CPU or CUDA(GPU)
        img_dir str: Folder, where need to save frames video
        video_dir str: Folder, where need to save video clips
        model_dir str: Folder, where save model for classification images
        recode: bool
        high_accuracy bool: Do I need to find a certain frame of transition between classes?
        rm_files list: list of bool value
        save_cls_tbl bool: save table with classification frames
        batch_size int: Batch size images for model
        ffmpeg_v str: Level verbose for ffmpeg calls
        ffmpeg FFMpeg: class FFMPeg for working with video
        model Optional[models.ResNet]: Model for classification images
        transformation Optional[transforms.Compose]: Transformations for images before passing to the model
        prediction Optional[np.ndarray]: An array predictions of model
    """

    def __init__(self,
                 path: str,
                 output_name: str,
                 skip_time: int = 1,
                 second_step: float = 1,
                 device: str = 'cpu',
                 img_dir: str = 'images',
                 video_dir: str = 'video',
                 model_dir: str = 'models',
                 logging: bool = False,
                 recode: bool = True,
                 high_accuracy: bool = True,
                 rm_tmp_files: List[bool] | bool = True,
                 save_cls_tbl: bool = False,
                 batch_size: int = 128,
                 ffmpeg_verbose: str = '-loglevel quiet -stats',
                 model: Optional[models.ResNet] = None,
                 transformation: Optional[transforms.Compose] = None,
                 prediction: Optional[np.ndarray] = None,
                 **kwargs):
        self.path = path
        self.output_name = output_name
        self.skip_time = skip_time
        self.second_step = second_step
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device != 'cpu' else device
        self.img_dir = img_dir
        self.video_dir = video_dir
        self.model_dir = model_dir
        self.logging = logging
        self.recode = recode
        self.high_accuracy = high_accuracy
        self.rm_files = rm_tmp_files if isinstance(rm_tmp_files, List) else [rm_tmp_files] * 2
        self.save_cls_tbl = save_cls_tbl
        self.batch_size = batch_size
        self.ffmpeg_v = ffmpeg_verbose
        self.ffmpeg = FFMpeg(
            path=self.path,
            img_dir=self.img_dir,
            video_dir=self.video_dir,
            output_name=self.output_name,
            device=self.device,
            verbose_mode=self.ffmpeg_v,
            second_step=self.second_step,
            logging=self.logging,
            recode=self.recode,
            rm_tmp_image=self.rm_files[0],
            rm_tmp_video=self.rm_files[1],
            **kwargs
        )
        self.model = model
        self.transformation = transformation
        self.prediction = prediction
        self.kwargs = kwargs

    def __repr__(self):
        return f"""ExtractorBroadcast(
            "skip_time": {self.skip_time},
            "second_step": {self.second_step},
            "device": {self.device},
            "batch_size": {self.batch_size},
            "logging": {self.logging},
            "recode": {self.recode},
            "high_accuracy": {self.high_accuracy}
        )"""

    @time_complete(text="Общее время работы программы:")
    def main_camera_video(self) -> None:
        """

        Returns:

        """
        if self.logging:
            print(self)

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

    @time_complete(text="Инициализация модели:")
    def init_model(self) -> models.ResNet:
        """
        Initialization model for classification

        Returns:
            models.ResNet: model for classification

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

    @time_complete(text="Классификация кадров: ")
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

    @time_complete(text="Общее время классификации кадров видео:")
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

    @time_complete(text="Создания таблицы видео интервалов:")
    def create_time_information(self, prediction: np.ndarray) -> ExtractorDF:
        """

        Args:
            prediction:

        Returns:

        """
        data = ExtractorDF(prediction)
        data.img_classification_df(self.second_step, self.ffmpeg.fps)
        if self.save_cls_tbl:
            tbl = (
                data.df
                .assign(
                    game_name = self.kwargs.get("game_name") if not self.kwargs.get("game_name") is None else Path(self.path).stem,
                    link_on_source = self.kwargs.get("link_on_source", ""),
                    probability = np.max(softmax(data.prediction, axis=1), axis=1)
                )
                .drop(
                    columns = ["mark_0", "mark_1", "prev_value", "next_value"]
                )
                .pipe(lambda df_: df_.loc[:, ["game_name", "link_on_source", "mark",
                                              "probability", "real_time"]])
            )
            if self.kwargs.get("cls_save_path", None) is not None:
                tbl.to_csv(self.kwargs.get("cls_save_path"), index=False)
            else:
                tbl.to_csv("classification_tbl.csv", index=False)
        data.main_camera_parts(self.skip_time)

        return data

    @time_complete(text="Общее время поиска кадров перехода:")
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
        screen_frame = 24 if self.second_step == 1 else 49
        skip_first_sec = 1 if self.second_step == 4 else 0
        frames = np.array([(screen_frame + skip_first_sec) + (self.ffmpeg.fps * self.second_step * i) for i in
                           range(np.floor(self.ffmpeg.num_frame / self.ffmpeg.fps / self.second_step).astype(np.int16))])

        st_fr = data.df.start_index.tolist()
        st_frames = np.array([frames[idx] for idx in st_fr])
        left_frames = np.clip(
            np.array([np.arange(st_frame - (self.ffmpeg.fps * self.second_step) + 1, st_frame) for st_frame in st_frames]),
            a_min=0,
            a_max=self.ffmpeg.num_frame - (100 / self.ffmpeg.fps)
        )

        end_fr = data.df.end_index.tolist()
        end_frames = np.array([frames[idx] for idx in end_fr])
        right_frames = np.clip(
            np.array([np.arange(st_frame + 1, st_frame + (self.ffmpeg.fps * self.second_step)) for st_frame in end_frames]),
            a_min=0,
            a_max=self.ffmpeg.num_frame - (100 / self.ffmpeg.fps)
        )

        new_time = self.step_frames(left_frames, right_frames, model, transformation)

        data.upd_main_camera(new_time[::2], new_time[1::2])

        return data

    @time_complete(text="Нарезка видеофрагментов:")
    def ffmpeg_cut_videos(self, data: ExtractorDF) -> None:
        """

        Args:
            data:
            recode:

        Returns:

        """
        for i, row in enumerate(data.df.itertuples()):
            self.ffmpeg.cut_videos(row.start_time, row.duration, i)

        with open(f'{self.video_dir}/file.txt', 'w', encoding='utf-8') as file_desc:
            for i in range(data.df.shape[0]):
                file_desc.write(f"file 'video_{i}.mkv'\n")
