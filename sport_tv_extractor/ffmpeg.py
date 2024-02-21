import re
import subprocess
from pathlib import Path

import numpy as np


class FFMpeg(object):

    def __init__(self,
                 path: str,
                 img_dir: str,
                 video_dir: str,
                 output_name: str,
                 device: str,
                 verbose_mode: str = '-loglevel quiet -stats',
                 rm_tmp_image: bool = True,
                 rm_tmp_video: bool = True
                 ):
        self.path = path
        self.img_dir = img_dir
        self.video_dir = video_dir
        self.output_name = output_name
        self.device = device
        self.verbose = verbose_mode
        self.rm_img = rm_tmp_image
        self.rm_video = rm_tmp_video
        self.codec = self._get_codec()
        self.num_frame = None
        self.fps = None
        self.bitrate = None
        Path.cwd().joinpath(self.img_dir).mkdir(parents=True, exist_ok=True)
        Path.cwd().joinpath(self.video_dir).mkdir(parents=True, exist_ok=True)

    def _get_codec(self):
        if self.device == 'cpu':
            return ['', '-c:v libx264']
        else:
            return ['-c:v h264_cuvid', '-c:v h264_cuvid']

    def cut_frames(self) -> None:
        cmd = f'ffmpeg {self.codec[0]} -i {self.path} -vf fps=1 -qscale:v 2 {self.verbose} {self.img_dir}/img-%02d.jpeg'
        self._call_subprocess(cmd)

    def loop_cut_frames(self, img_dir, arr_frame: np.ndarray) -> None:
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
        subprocess.call(cmd, shell=True, executable='/bin/bash')

    def cut_videos(self,
                   start_time: float,
                   duration: float,
                   idx_video: int
                   ) -> None:
        if self.device == 'cpu':
            cmd = f'ffmpeg -ss {start_time} -t {duration} -i {self.path} -vf "setpts=PTS-STARTPTS" {self.codec[1]} -crf 21 -preset ultrafast {self.verbose} -an {self.video_dir}/video_{idx_video}.mkv'
        else:
            cmd = f'ffmpeg {self.codec[1]} -hwaccel cuvid -ss {start_time} -t {duration} -i {self.path} -vf "setpts=PTS-STARTPTS" -c:v h264_nvenc -b:v 3.4M -preset "hq" {self.verbose} -an -y {self.video_dir}/video_{idx_video}.mkv'
        self._call_subprocess(cmd)

    def concat_videos(self) -> None:
        cmd = f'ffmpeg -f concat -safe 0 -i {self.video_dir}/file.txt -c:v copy {self.verbose} {self.output_name}'
        self._call_subprocess(cmd)

    def rm_tmp_files(self) -> None:
        if self.rm_img:
            cmd = f'rm -rf {self.img_dir}'
            self._call_subprocess(cmd)
        if self.rm_video:
            cmd = f'rm -rf {self.video_dir}'
            self._call_subprocess(cmd)

    def get_num_frame(self):
        if self.num_frame is None:
            p = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams',
                                  'v:0', '-count_packets', '-show_entries',
                                  'stream=nb_read_packets', '-of', 'csv=p=0',
                                  self.path.replace('\ ', ' ')],
                                 stdout=subprocess.PIPE)
            out, err = p.communicate()
            self.num_frame = int(out.decode('utf-8').strip('\n'))
        return self.num_frame

    def get_fps(self):
        if self.fps is None:
            p = subprocess.Popen(['ffprobe', '-v', 'error', '-select_streams',
                                  'v:0', '-show_entries', 'stream=avg_frame_rate',
                                  '-of', 'default=noprint_wrappers=1:nokey=1',
                                  self.path.replace('\ ', ' ')],
                                 stdout=subprocess.PIPE)
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