import re
import subprocess
from pathlib import Path

import numpy as np


class FFMpeg(object):

    def __init__(self,
                 path: str,
                 img_folder: str,
                 video_folder: str,
                 output_name: str,
                 verbose_mode: str = '-loglevel quiet -stats',
                 rm_tmp_image: bool = True,
                 rm_tmp_video: bool = True
                 ):
        self.path = path
        self.img_folder = img_folder
        self.video_folder = video_folder
        self.output_name = output_name
        self.verbose = verbose_mode
        self.rm_img = rm_tmp_image
        self.rm_video = rm_tmp_video
        self.num_frame = None
        self.fps = None
        self.bitrate = None
        Path.cwd().joinpath(self.img_folder).mkdir(parents=True, exist_ok=True)
        Path.cwd().joinpath(self.video_folder).mkdir(parents=True, exist_ok=True)

    def cut_frames(self) -> None:
        command = f'ffmpeg -c:v h264_cuvid -i {self.path} -vf fps=1 -qscale:v 2 {self.verbose} {self.img_folder}/img-%02d.jpeg'
        # command = f'ffmpeg -i {self.path} -vf fps=1 -qscale:v 2 {self.verbose} {self.img_folder}/img-%02d.jpeg'
        self._call_subprocess(command)

    def loop_cut_frames(self, img_dir, arr_frame: np.ndarray) -> None:
        command = "".join([
            f'arr=({" ".join([str(x) for x in arr_frame])}); '
            'for i in "${arr[@]}"; do ffmpeg -ss "$i" -i ',
            self.path,
            ' -frames:v 1 -qscale:v 2 -loglevel panic -hide_banner ',
            f'{str(img_dir)}/',
            '"img-${i%.*}".jpeg; done;'
        ])
        subprocess.call(command, shell=True, executable='/bin/bash')

    def cut_videos(self,
                   start_time: float,
                   duration: float,
                   idx_video: int
                   ) -> None:
        command = f'ffmpeg -c:v h264_cuvid -hwaccel cuvid -ss {start_time} -t {duration} -i {self.path} -vf "setpts=PTS-STARTPTS" -c:v h264_nvenc -b:v 5M -preset "hp" {self.verbose} -an -y {self.video_folder}/video_{idx_video}.mkv'
        # command = f'ffmpeg -ss {start_time} -t {duration} -i {self.path} -vf "setpts=PTS-STARTPTS" -c:v libx264 -crf 27 -preset ultrafast {self.verbose} -an {self.video_folder}/video_{idx_video}.mkv'
        self._call_subprocess(command)

    def concat_videos(self) -> None:
        command = f'ffmpeg -f concat -safe 0 -i {self.video_folder}/file.txt -c:v copy {self.verbose} {self.output_name}'
        self._call_subprocess(command)

    def rm_tmp_files(self) -> None:
        if self.rm_img:
            command = f'rm -rf {self.img_folder}'
            self._call_subprocess(command)
        if self.rm_video:
            command = f'rm -rf {self.video_folder}'
            self._call_subprocess(command)

    @property
    def get_num_frame(self):
        if self.num_frame is None:
            p = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams',
                                  'v:0', '-count_packets', '-show_entries',
                                  'stream=nb_read_packets', '-of', 'csv=p=0', self.path], stdout=subprocess.PIPE)
            out, err = p.communicate()
            self.num_frame = int(out.decode('utf-8').strip('\n'))
        return self.num_frame

    @property
    def get_fps(self):
        if self.fps is None:
            p = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams',
                                  'v:0', '-show_entries', 'stream=avg_frame_rate',
                                  '-of', 'default=noprint_wrappers=1:nokey=1', self.path], stdout=subprocess.PIPE)
            out, err = p.communicate()
            self.fps = int(out.decode('utf-8').strip('\n').split('/')[0])
        return self.fps

    def bitrate_video(self):
        if self.bitrate is None:
            self._call_subprocess(f'ffprobe {self.path} 2>meta_info.txt')
            with Path('meta_info.txt').open(mode='r') as f:
                meta = f.readlines()
            v_bt = [re.search(r'(?<=bitrate:\s)\d+\s+.+(?=\n)', line).group() for line in meta if
                    re.search(r'bitrate:\s+\d+\s+.+(?=\n)', line) is not None]
            a_bt = [re.search(r'\d+ \w+\/\w+', line).group() for line in meta if
                    re.search(r'Audio', line) is not None]

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
    def _call_subprocess(command: str) -> None:
        subprocess.call(command, shell=True)
