import re
import subprocess as sp
from pathlib import Path
from typing import Optional, List

import numpy as np

from utils import time_complete


class FFMpeg(object):
    """ Class for work with ffmpeg

    The class allows you to process video using ffmpeg calls:
    1. Get frames
    2. Cut video clips
    3. Concat video clips into one video

    Attributes:
        path str: Path to video
        second_step float:
        img_dir str: Folder, where need to save frames video
        video_dir str: Folder, where need to save video clips
        output_name str: Name for output file
        device str: Device on which must make video processing: CPU or CUDA(GPU)
        verbose str: Level verbose for ffmpeg calls
        logging bool:
        recode bool:
        rm_img bool: Delete temporary frames?
        rm_video bool: Delete temporary video clips?
        codec List[str]:
        num_frame Optional[int]:
        fps Optional[int]:
        bitrate Optional[float]: 
    """

    def __init__(self,
                 path: str,
                 img_dir: str,
                 video_dir: str,
                 output_name: str,
                 device: str,
                 verbose_mode: str = '-loglevel quiet -stats',
                 second_step: float = 1,
                 logging: bool = False,
                 recode: bool = True,
                 rm_tmp_image: bool = True,
                 rm_tmp_video: bool = True,
                 **kwargs
                 ):
        self.path = path
        self.img_dir = img_dir
        self.video_dir = video_dir
        self.output_name = output_name
        self.device = device
        self.verbose = verbose_mode
        self.second_step = second_step
        self.logging = logging
        self.recode = recode
        self.rm_img = rm_tmp_image
        self.rm_video = rm_tmp_video
        self.codec: List[str] = self._get_codec() if kwargs.get("codec", None) is None else kwargs.get("codec")
        self.num_frame: Optional[int] = kwargs.get("num_frame", None)
        self.fps: Optional[int] = kwargs.get("fps", None)
        self.bitrate: Optional[float] = kwargs.get("bitrate", None)
        Path.cwd().joinpath(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path.cwd().joinpath(self.video_dir).mkdir(parents=True, exist_ok=True)

    def _get_codec(self):
        """

        Returns:

        """
        if self.device == 'cpu':
            return ['', '-c:v libx264']
        else:
            return ['-c:v h264_cuvid', '-c:v h264_cuvid']

    @time_complete(text="Получение кадров видео с шагом 1 секунда:")
    def cut_frames(self) -> None:
        """

        Returns:

        """
        cmd = f'ffmpeg {self.codec[0]} -i {self.path} -vf "fps={1/self.second_step}, scale=224:224" -qscale:v 2 {self.verbose} {self.img_dir}/img-%02d.jpeg'
        self._call_sp(cmd)

    @time_complete(text="Получение кадров видео в цикле:")
    def loop_cut_frames(self, img_dir: str, arr_frame: np.ndarray) -> None:
        """

        Args:
            img_dir:
            arr_frame:

        Returns:

        """
        cmd = "".join([
            f'arr=({" ".join([str(x) for x in arr_frame])}); '
            'for i in "${arr[@]}"; do ffmpeg ',
            self.codec[0],
            ' -ss "$i" -i ',
            self.path,
            ' -frames:v 1 -qscale:v 2 -loglevel panic -hide_banner ',
            f'{str(img_dir)}/',
            '"img-${i%.*}".jpeg; done;'
        ])
        sp.call(cmd, shell=True, executable='/bin/bash')

    def cut_videos(self,
                   start_time: float,
                   duration: float,
                   idx_video: int,
                   ) -> None:
        """

        Args:
            start_time:
            duration:
            idx_video:
            recode:

        Returns:

        """
        if self.recode:
            if self.device == 'cpu':
                cmd = f'ffmpeg -ss {start_time} -t {duration} -i {self.path} -vf "setpts=PTS-STARTPTS" {self.codec[1]} -crf 21 -preset ultrafast {self.verbose} -an {self.video_dir}/video_{idx_video}.mkv'
            else:
                cmd = f'ffmpeg {self.codec[1]} -ss {start_time} -t {duration} -i {self.path} -vf "setpts=PTS-STARTPTS" -c:v h264_nvenc -b:v {self.bitrate_video()}M -preset "hq" {self.verbose} -an -y {self.video_dir}/video_{idx_video}.mkv'
        else:
            cmd = f'ffmpeg -ss {start_time} -t {duration} -i {self.path} -c:v copy -an {self.verbose} {self.video_dir}/video_{idx_video}.mkv'
        self._call_sp(cmd)

    @time_complete(text="Объединение видеофрагментов в один файл:")
    def concat_videos(self) -> None:
        """

        Returns:

        """
        cmd = f'ffmpeg -f concat -safe 0 -i {self.video_dir}/file.txt -c:v copy {self.verbose} {self.output_name}'
        self._call_sp(cmd)

    def rm_tmp_files(self) -> None:
        """

        Returns:

        """
        if self.rm_img:
            cmd = f'rm -rf {self.img_dir}'
            self._call_sp(cmd)
        if self.rm_video:
            cmd = f'rm -rf {self.video_dir}'
            self._call_sp(cmd)

    @time_complete(text="Получение количества кадров в видео:")
    def get_num_frame(self) -> int:
        """

        Returns:

        """
        if self.num_frame is None:
            p = sp.Popen(['ffprobe', '-v', 'error', '-select_streams',
                                  'v:0', '-count_packets', '-show_entries',
                                  'stream=nb_read_packets', '-of', 'csv=p=0',
                                  self.path.replace('\ ', ' ')],
                                 stdout=sp.PIPE)
            out, err = p.communicate()
            self.num_frame = int(out.decode('utf-8').strip('\n'))
        return self.num_frame

    @time_complete(text="Получение FPS видео:")
    def get_fps(self) -> int:
        """

        Returns:

        """
        if self.fps is None:
            p = sp.Popen(['ffprobe', '-v', 'error', '-select_streams',
                                  'v:0', '-show_entries', 'stream=avg_frame_rate',
                                  '-of', 'default=noprint_wrappers=1:nokey=1',
                                  self.path.replace('\ ', ' ')],
                                 stdout=sp.PIPE)
            out, err = p.communicate()
            self.fps = int(out.decode('utf-8').strip('\n').split('/')[0])
        return self.fps

    def bitrate_video(self) -> float:
        """

        Returns:

        """
        if self.bitrate is None:

            cmd = ['ffprobe', '-i', self.path.replace('\ ', ' ')]
            pipe = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 5)
            meta = pipe.stderr.read().decode('utf-8').split('\n')
            v_bt = [re.search(r'(?<=bitrate:\s)\d+\s+.+', line).group() for line in meta if
                    re.search(r'bitrate:\s+\d+\s+.+', line) is not None]
            try:
                a_bt = [re.search(r'\d+ \w+\/\w+', line).group() for line in meta if
                        re.search(r'Audio', line) is not None]
            except AttributeError:
                a_bt = ['0 kb/s']

            unit = v_bt[0].split(' ')[1]

            if all([unit == audio.split(' ')[1] for audio in a_bt]):
                full_video = float(v_bt[0].split(' ')[0])
                full_audio = sum([float(audio.split(' ')[0]) for audio in a_bt])
                self.bitrate = np.round((full_video - full_audio) / 1000, 2)
                return self.bitrate
            else:
                raise ValueError
        else:
            return self.bitrate

    @staticmethod
    def _call_sp(command: str) -> None:
        """

        Args:
            command:

        Returns:

        """
        sp.call(command, shell=True)


def check_cut_frames(path: str, step: int = 1, n_second: int = 12) -> dict[str: list[int | float]]:
    fps_cmd = sp.Popen(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=avg_frame_rate', '-of',
                        'default=noprint_wrappers=1:nokey=1',
                        path], stdout=sp.PIPE)
    out, err = fps_cmd.communicate()
    fps = int(out.decode('utf-8').strip('\n').split('/')[0])

    command = ['ffmpeg',
               '-ss', '00:00:00',
               '-t', str(n_second),
               '-c:v', 'h264_cuvid',
               '-i', path,
               '-vf', 'scale=224:224',
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-loglevel', 'error',
               '-hide_banner',
               '-vcodec', 'rawvideo',
               '-']
    pipe_every_frame = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)

    raw_images = pipe_every_frame.stdout.read(n_second * fps * 224 * 224 * 3)
    pipe_every_frame.stdout.flush()
    every_frames = np.frombuffer(raw_images, dtype='uint8').reshape((n_second * fps, 224, 224, 3))

    command[10] = f'fps={1 / step}, scale=224:224'

    pipe_step = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)

    step_raw_images = pipe_step.stdout.read(int(np.floor(n_second / step) * 224 * 224 * 3))
    pipe_step.stdout.flush()
    step_frames = np.frombuffer(step_raw_images, dtype='uint8').reshape((int(np.floor(n_second / step)), 224, 224, 3))

    d = {"num_frame": [], "num_frame_in_sec": [], "time_video": []}
    cnt = 0
    for i, frame in enumerate(every_frames):
        try:
            diff = np.sum(frame - step_frames[cnt])
        except IndexError:
            break
        if diff == 0:
            d["num_frame"].append(i)
            d["num_frame_in_sec"].append(i % fps)
            d["time_video"].append((i // fps) + (i % fps * 2 / 100))
            cnt += 1
    return d
