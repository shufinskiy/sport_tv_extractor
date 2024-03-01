from pathlib import Path
import shutil
from typing import List

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
        rm_tmp
        ffmpeg_v
    """

    def __init__(self,
                 path,
                 output_name,
                 skip_time=1,
                 device='cpu',
                 img_dir='images',
                 video_dir='video',
                 model_dir='models',
                 high_accuracy=True,
                 rm_tmp_files=True,
                 batch_size=128,
                 ffmpeg_verbose='-loglevel quiet -stats'):
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

    def main_camera_video(self):

        model = self.init_model()

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.ffmpeg.cut_frames()
        prediction = self.classification_images(model, val_transforms)

        self.ffmpeg.get_fps()
        self.ffmpeg.get_num_frame()

        data = self.create_time_information(prediction)

        if self.high_accuracy:
            self.time_high_accuracy(data, model, val_transforms)

        for i, row in enumerate(data.df.itertuples()):
            self.ffmpeg_cut_videos(i, row, data)

        self.ffmpeg.concat_videos()

        self.ffmpeg.rm_tmp_files()

    def download_model_dict(self, url=URL_RESNET, progress=False):
        return torch.hub.load_state_dict_from_url(url=url, model_dir=self.model_dir, progress=progress)

    def init_model(self):
        model = models.resnet18()
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(self.download_model_dict())
        model.eval()
        model.to(self.device)

        return model

    def prediction_image(self, for_pred, model, transformation, tmp_dir=TMP_DIR):
        path = Path(tmp_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.ffmpeg.loop_cut_frames(path, for_pred)

        dataset = CustomImageFolder(path, transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = self.class_prediction(model, dataloader, shape=len(dataset))
        shutil.rmtree(path)
        return prediction

    def step_frames(self, left_frames, right_frames, model, transformation, tmp_dir=TMP_DIR):

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

    def class_prediction(self, model, dataloader, shape):
        prediction = np.empty(shape)
        for i, inputs in enumerate(dataloader):
            with torch.set_grad_enabled(False):
                inputs = inputs.to(self.device)
                preds = model(inputs).cpu()
                preds = preds.argmax(dim=1) if isinstance(shape, int) else preds.numpy()
                prediction[i * self.batch_size:(i + 1) * self.batch_size] = preds

        return prediction

    def classification_images(self, model, transformation):
        dataset = CustomImageFolder(self.img_dir, transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = self.class_prediction(model, dataloader, shape=(len(dataset), 2))

        return prediction

    def create_time_information(self, prediction):
        data = ExtractorDF(prediction)
        data.img_classification_df(self.ffmpeg.fps)
        data.main_camera_parts(self.skip_time)

        return data

    def time_high_accuracy(self, data, model, transformation):
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

    def ffmpeg_cut_videos(self, i, row, data):
        self.ffmpeg.cut_videos(row.start_time, row.duration, i)

        with open(f'{self.video_dir}/file.txt', 'w', encoding='utf-8') as file_desc:
            for i in range(data.df.shape[0]):
                file_desc.write(f"file 'video_{i}.mkv'\n")


if __name__ == "__main__":
    from time import time

    # PATH_TO_VIDEO = '/home/shuf91/Загрузки/25.02.2024.Serie A. Milan - Atalanta.mkv'.replace(' ', '\ ')
    PATH_TO_VIDEO = '/home/shuf91/env/video_sport_game/package/fio_int.mkv'
    IMADE_DIR = '/home/shuf91/env/video_sport_game/package/images'
    VIDEO_DIR = '/home/shuf91/env/video_sport_game/package/video'
    OUTPUT_NAME = '/home/shuf91/env/video_sport_game/package/fio_int_filt.mkv'
    MODEL_DIR = '/home/shuf91/env/video_sport_game/package/models'

    s = time()
    extractor = ExtractorBroadcast(
        path=PATH_TO_VIDEO,
        output_name=OUTPUT_NAME,
        device='cuda',
        skip_time=5,
        img_dir=IMADE_DIR,
        video_dir=VIDEO_DIR,
        model_dir=MODEL_DIR,
        high_accuracy=True
    )
    extractor.main_camera_video()
    f = time()
    print(f'{f-s}')
