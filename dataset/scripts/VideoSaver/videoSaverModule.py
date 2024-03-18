import yt_dlp
import sys
import ffmpeg
import json

def create_format_opts():
    pass

class VideoSaver:
    
    
    def __init__(self, opts):
        self.opts = opts
        self.ydl = yt_dlp.YoutubeDL(self.opts)
    
    def trim_full(self, input_path : str, output_path : str, start : int = 30, end : int = 60) -> None:
        input_stream = ffmpeg.input(input_path)

        vid = (
            input_stream.video
            .trim(start=start, end=end)
            .setpts('PTS-STARTPTS')
        )
        aud = (
            input_stream.audio
            .filter_('atrim', start=start, end=end)
            .filter_('asetpts', 'PTS-STARTPTS')
        )

        joined = ffmpeg.concat(vid, aud, v=1, a=1).node
        output = ffmpeg.output(joined[0], joined[1], output_path)
        output.run()

    def trim_part(self, video_path : str, audio_path : str, output_path : str, start : int = 0, end : int = 15) -> None:
        video_stream = ffmpeg.input(video_path)
        audio_stream = ffmpeg.input(audio_path)

        vid = (
            video_stream.video
            .trim(start=start, end=end)
            .setpts('PTS-STARTPTS')
        )
        aud = (
            audio_stream.audio
            .filter_('atrim', start=start, end=end)
            .filter_('asetpts', 'PTS-STARTPTS')
        )
        joined = ffmpeg.concat(vid, aud, v=1, a=1).node
        output = ffmpeg.output(joined[0], joined[1], output_path)
        output.run()

    def convert_time_to_secs(self, time : str):
        lst = time.split(':')
        pw = 1
        cur = 0
        for x in lst[::-1]:
            cur += pw * int(x)
            pw *= 60
        return cur
    
    def download(self, url : str, start_time : str, end_time : str, output_path : str, **kwargs):
        start = self.convert_time_to_secs(start_time)
        end = self.convert_time_to_secs(end_time)

        if 'opts' in kwargs:
            self.ydl = kwargs['opts']
        
        info = self.ydl.extract_info(url, download=False)

        if start < 0:
            start = 0
        if end >= info['duration']:
            end = info['duration']

        print(info['duration'])

        if 'requested_formats' in info:
            
            video_url = info['requested_formats'][0]['url']
            audio_url = info['requested_formats'][1]['url']

            self.trim_part(video_url, audio_url, output_path, start, end)
        else:
            if 'url' in info:
                full_url = info['url']
                self.trim_full(full_url, output_path, start, end)

        pass