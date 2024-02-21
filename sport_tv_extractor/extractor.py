from pathlib import Path
import shutil

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models, transforms

from ffmpeg import FFMpeg
from utils import CustomImageFolder, ExtractorDF

TMP_DIR = '/home/shuf91/env/video_sport_game/package/left_right_check/'


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
        self.high_accuracy = high_accuracy
        self.rm_tmp = rm_tmp_files
        self.batch_size = batch_size
        self.ffmpeg_v = ffmpeg_verbose
        self.ffmpeg = FFMpeg(
            path=self.path,
            img_dir=self.img_dir,
            video_dir=self.video_dir,
            output_name=self.output_name,
            device=self.device,
            verbose_mode=self.ffmpeg_v,
            rm_tmp_image=self.rm_tmp,
            rm_tmp_video=self.rm_tmp
        )

    def main_camera_video(self):
        self.ffmpeg.cut_frames()

        model = self._init_model()

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        dataset = CustomImageFolder(self.img_dir, val_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = np.empty((len(dataset), 2))
        for i, inputs in enumerate(dataloader):
            with torch.set_grad_enabled(False):
                inputs = inputs.to(self.device)
                preds = model(inputs)
                prediction[i * self.batch_size:(i + 1) * self.batch_size] = preds.cpu().numpy()

        self.ffmpeg.get_fps()
        self.ffmpeg.get_num_frame()

        data = ExtractorDF(prediction, dataset.paths)
        data.img_classification_df(self.ffmpeg.fps)
        data.main_camera_parts(self.skip_time)

        if self.high_accuracy:

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

            new_start_time = self.step_frames(left_frames, model, val_transforms, side='left')
            new_end_time = self.step_frames(right_frames, model, val_transforms, side='right')

            data.upd_main_camera(new_start_time, new_end_time)

        for i, row in enumerate(data.df.itertuples()):
            self.ffmpeg.cut_videos(row.start_time, row.duration, i)

        with open(f'{self.video_dir}/file.txt', 'w', encoding='utf-8') as f:
            for i in range(data.df.shape[0]):
                f.write(f"file 'video_{i}.mkv'\n")

        self.ffmpeg.concat_videos()

        self.ffmpeg.rm_tmp_files()

    def _init_model(self):
        model = models.resnet18()
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load('/home/shuf91/env/video_sport_game/package/football_checkpoint6.pt'))
        model.eval()
        model.to(self.device)

        return model

    def prediction_image(self, for_pred, model, transformation, tmp_dir=TMP_DIR):
        path = Path(tmp_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.ffmpeg.loop_cut_frames(path, for_pred)

        dataset = CustomImageFolder(path, transformation)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = np.empty(len(dataset), )
        for i, inputs in enumerate(dataloader):
            with torch.set_grad_enabled(False):
                inputs = inputs.to(self.device)
                preds = model(inputs)
                prediction[i * self.batch_size:(i + 1) * self.batch_size] = preds.cpu().argmax(dim=1)
        shutil.rmtree(path)
        return prediction

    def step_frames(self, step_frames, model, transformation, side, tmp_dir=TMP_DIR):

        if step_frames.shape[1] == 1:
            for_pred = step_frames.reshape(-1, ) / self.ffmpeg.fps
            prediction = self.prediction_image(for_pred, model, transformation, tmp_dir)

            next_frame = 1 if side == 'left' else -1
            final_frame = np.array(
                [t if prediction[i] == 0 else t + next_frame for i, t in enumerate(step_frames.reshape(-1, ))])
            return final_frame / self.ffmpeg.fps
        m = step_frames.shape[1] // 2
        low = m
        high = m + 1 if step_frames.shape[1] % 2 == 1 else m

        for_pred = step_frames[:, m] / self.ffmpeg.fps
        prediction = self.prediction_image(for_pred, model, transformation, tmp_dir)
        check_class = 0 if side == 'left' else 1

        step_frames = np.vstack([t[:low] if prediction[i] == check_class else t[high:]
                                 for i, t in enumerate(step_frames)])
        return self.step_frames(step_frames, model, transformation, tmp_dir)


if __name__ == "__main__":
    from time import time

    PATH_TO_VIDEO = '/home/shuf91/Загрузки/20.02.2024.ChLeague.Inter Milan - Club Atletico de Madrid.mkv'.replace(' ', '\ ')
    IMADE_DIR = '/home/shuf91/env/video_sport_game/package/images'
    VIDEO_DIR = '/home/shuf91/env/video_sport_game/package/video'
    OUTPUT_NAME = '/home/shuf91/env/video_sport_game/package/int_atm_5.mkv'

    s = time()
    extractor = ExtractorBroadcast(
        path=PATH_TO_VIDEO,
        output_name=OUTPUT_NAME,
        device='cuda',
        skip_time=5,
        img_dir=IMADE_DIR,
        video_dir=VIDEO_DIR,
        high_accuracy=True
    )
    extractor.main_camera_video()
    f = time()
    print(f'{f-s}')
