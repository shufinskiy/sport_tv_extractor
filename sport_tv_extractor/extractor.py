from pathlib import Path
import shutil
from typing import List
import subprocess as sp

import numpy as np
# import pandas as pd

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

        prediction = self.class_prediction(model, dataloader, shape=(len(dataset), 2))

        self.ffmpeg.get_fps()
        self.ffmpeg.get_num_frame()

        data = ExtractorDF(prediction)
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

    def main_camera_video1(self):
        self.ffmpeg_cut_frames()

        model = self._init_model()

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        prediction, paths = self.step2(model, val_transforms)

        self.ffmpeg.get_fps()
        self.ffmpeg.get_num_frame()

        data = self.step3(prediction, paths)

        if self.high_accuracy:
            data = self.step4(data, model, val_transforms)

        self.step5(data)
        self.step6()
        self.ffmpeg.rm_tmp_files()

    def download_model_dict(self, url=URL_RESNET, progress=False):
        return torch.hub.load_state_dict_from_url(url=url, model_dir=self.model_dir, progress=progress)

    def _init_model(self):
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
        return self.step_frames(step_frames, model, transformation, side, tmp_dir)

    def class_prediction(self, model, dataloader, shape):
        prediction = np.empty(shape)
        for i, inputs in enumerate(dataloader):
            with torch.set_grad_enabled(False):
                inputs = inputs.to(self.device)
                preds = model(inputs).cpu()
                preds = preds.argmax(dim=1) if isinstance(shape, int) else preds.numpy()
                prediction[i * self.batch_size:(i + 1) * self.batch_size] = preds

        return prediction

    def ffmpeg_cut_frames(self):
        self.ffmpeg.cut_frames()

    def step2(self, model, transforms):

        dataset = CustomImageFolder(self.img_dir, transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        prediction = np.empty((len(dataset), 2))
        for i, inputs in enumerate(dataloader):
            with torch.set_grad_enabled(False):
                inputs = inputs.to(self.device)
                preds = model(inputs)
                prediction[i * self.batch_size:(i + 1) * self.batch_size] = preds.cpu().numpy()

        return prediction, dataset.paths

    def step3(self, prediction):
        data = ExtractorDF(prediction)
        data.img_classification_df(self.ffmpeg.fps)
        data.main_camera_parts(self.skip_time)

        return data

    def step4(self, data, model, transforms):
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

        new_start_time = self.step_frames(left_frames, model, transforms, side='left')
        new_end_time = self.step_frames(right_frames, model, transforms, side='right')

        data.upd_main_camera(new_start_time, new_end_time)

        return data

    def step5(self, data):
        for i, row in enumerate(data.df.itertuples()):
            self.ffmpeg.cut_videos(row.start_time, row.duration, i)

        with open(f'{self.video_dir}/file.txt', 'w', encoding='utf-8') as f:
            for i in range(data.df.shape[0]):
                f.write(f"file 'video_{i}.mkv'\n")

    def step6(self):
        self.ffmpeg.concat_videos()


class BufferExtractorBroadcast(ExtractorBroadcast):

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
        super().__init__(path=path,
                         output_name=output_name,
                         skip_time=skip_time,
                         device=device,
                         img_dir=img_dir,
                         video_dir=video_dir,
                         model_dir=model_dir,
                         high_accuracy=high_accuracy,
                         rm_tmp_files=rm_tmp_files,
                         batch_size=batch_size,
                         ffmpeg_verbose=ffmpeg_verbose)

    def buffer_main_camera_video(self):
        s1 = time()
        num_frames = self.ffmpeg.get_num_frame()
        fps = self.ffmpeg.get_fps()
        cnt_frames = int(np.modf(num_frames / fps)[1] + (1 if np.modf(num_frames / fps)[0] > 0.5 else 0))
        command = ['ffmpeg',
                   '-c:v', 'h264_cuvid',
                   '-i', self.path.replace('\ ', ' '),
                   '-vf', 'fps=1, scale=224:224',
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-loglevel', 'error',
                   '-hide_banner',
                   '-vcodec', 'rawvideo',
                   '-']
        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 9)
        frames_size = 512

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = models.resnet18()
        num_ftrs = model.fc.in_features

        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load('/home/shuf91/env/video_sport_game/package/models/main_camera_extractor.pt'))
        model.eval()
        model.to(device)

        prediction = np.empty((cnt_frames, 2))
        cnt = 0
        f1 = time()
        print(f'Время работы подготовительной части: {f1-s1}')
        while cnt*frames_size < cnt_frames:
            first_dim = np.min([frames_size, cnt_frames - (cnt*frames_size)])
            s2 = time()
            raw_image = pipe.stdout.read(first_dim * 224 * 224 * 3)
            if first_dim < frames_size:
                pipe.stdout.flush()
            f2 = time()
            print(f'Чтение из буффера {f2-s2}')
            s3 = time()
            image = np.frombuffer(raw_image, dtype='uint8')
            del raw_image
            image = image.reshape((first_dim, 224, 224, 3))

            img_tensor = torch.permute(torch.FloatTensor(image / 255), (0, 3, 1, 2))
            del image

            img_tensor[:, 0] = (img_tensor[:, 0] - 0.485) / 0.229
            img_tensor[:, 1] = (img_tensor[:, 1] - 0.456) / 0.224
            img_tensor[:, 2] = (img_tensor[:, 2] - 0.406) / 0.225

            cnt_img = img_tensor.size()[0]
            batch_size = 128
            for i in range(int(np.ceil(cnt_img / batch_size))):
                inputs = img_tensor[batch_size * i:batch_size * (i + 1)]
                with torch.set_grad_enabled(False):
                    inputs = inputs.to(device)
                    preds = model(inputs).cpu().numpy()
                    # preds = preds.argmax(dim=1)
                    prediction[(cnt * frames_size) + (i * batch_size):(cnt * frames_size) + (i + 1) * batch_size] = preds
            cnt += 1
            f3 = time()
            print(f'Остальная часть {f3-s3}')

        n = np.arange(prediction.shape[0]) + 0.48
        data = ExtractorDF(prediction)
        data.img_classification_df(self.ffmpeg.fps)
        data.main_camera_parts(self.skip_time)
        print(data.df.head())
        return prediction


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
    # pred = extractor.buffer_main_camera_video()
    extractor.main_camera_video()
    # extractor.ffmpeg_cut_frames()
    # extractor._init_model()
    # extractor.ffmpeg.bitrate_video()
    # extractor.step1()
    # model = extractor._init_model()
    # val_transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # pred, paths = extractor.step2(model, val_transforms)

    # np.save('/home/shuf91/env/video_sport_game/package/pred.npy', pred)
    # pd.Series(paths).to_csv('/home/shuf91/env/video_sport_game/package/paths.csv', index=False)
    # pred = np.load('/home/shuf91/env/video_sport_game/package/pred.npy')
    # paths = pd.read_csv('/home/shuf91/env/video_sport_game/package/paths.csv')
    #
    # extractor.ffmpeg.get_fps()
    # extractor.ffmpeg.get_num_frame()
    #
    # data = extractor.step3(pred, paths)
    # data = extractor.step4(data, model, val_transforms)
    f = time()
    print(f'{f-s}')
