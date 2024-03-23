import speech_recognition as s_r
import yt_dlp
from beetLibrary import convert_time_to_secs, read_table
import ffmpeg
import pandas as pd
import os
from os import path
from tqdm import tqdm
import time

class DataCollector:

    def __init__(self, opts):
        self.ydl = yt_dlp.YoutubeDL(opts)
        self.recognizer = s_r.Recognizer()
        pass

    def get_vid_aud_urls(self, url : str, **kwargs) -> tuple[str, str]:
        if 'opts' in kwargs:
            old_params = self.ydl.params.copy()
            self.ydl.params = kwargs['opts']
        
        info = self.ydl.extract_info(url, download=False)

        if 'opts' in kwargs:
            self.ydl.params = old_params

        if 'requested_formats' in info:
            
            video_url = info['requested_formats'][0]['url']
            audio_url = info['requested_formats'][1]['url']

            return video_url, audio_url
        else:
            if 'url' in info:
                full_url = info['url']
                return (full_url,)
            else:
                raise NameError('Something went wrong, while extracting URLS')
    
    def process_urls(self, video_url : str, audio_url : str, start : int, end : int):
        if audio_url == None:
            input_stream = ffmpeg.input(video_url)

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
        else:
            video_stream = ffmpeg.input(video_url)
            audio_stream = ffmpeg.input(audio_url)

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

        return vid, aud


    def get_text(self, audiofile_path : str) -> str:
        audio_path = path.join(path.dirname(path.realpath(__file__)), audiofile_path)
        sample = s_r.AudioFile(audio_path)
        print(sample.DURATION)

        with sample as audio:
            
            self.recognizer.adjust_for_ambient_noise(audio)
            content = self.recognizer.record(audio)
            return self.recognizer.recognize_google(content, language='ru-RU')

    def collect_sample(self, url : str, start_time : str, end_time : str, id : int, *, save : bool = False, output_directory : str = "data/") -> str:
        """
        Do: По заданной ссылке на youtube.com или другом хостинге выделяет фрагмент видео, указанный через start_time и end_time и скачивает отедельно видео и аудио в указанную директорию.
        Также в отдельном текстовом файле сохраняет текст, сказанный в этом фрагменте.
        Arguments:
        Return: возвращает транскрипцию видео, то есть текст, сказанный в нём.
        """

        os.makedirs(f'{output_directory}/{id}/', exist_ok=True)
        dir_template = f'{output_directory}/{id}/{id}'

        
        if (f'{id}.wav' in os.listdir(f'{output_directory}/{id}/')):
            text = self.get_text(dir_template + '.wav')
            return text

        start = convert_time_to_secs(start_time)
        end = convert_time_to_secs(end_time)

        video_url, audio_url = self.get_vid_aud_urls(url)

        vid, aud = self.process_urls(video_url, audio_url, start, end)
        

        print(os.listdir(f'{output_directory}/{id}/'))

        if not (dir_template + 'wav') in os.listdir(f'{output_directory}/{id}/'):
            output_audio = ffmpeg.output(aud, dir_template + '.wav')
            ffmpeg.run(output_audio, overwrite_output=True)

        with open(dir_template + '.txt', 'w') as text_file:
            text = self.get_text(dir_template + '.wav')
            text_file.write(text)

        if save:
            if not (dir_template + 'mp4') in os.listdir(f'{output_directory}/{id}/'):
                """
                optionally - video only
                """
                joined = ffmpeg.concat(vid, aud, v=1, a=1).node
                output_video = ffmpeg.output(joined[0], joined[1], dir_template + '.mp4')
                ffmpeg.run(output_video, overwrite_output=True)

        return text
    

    def collect_samples(self, data, *, save : bool = False, output_directory : str = "data/"):
        os.makedirs(f'{output_directory}/', exist_ok=True)

        table = read_table(data)


        for index, row in tqdm(table.iterrows()):
            id, url, start_time, end_time, text = row.values
            self.collect_sample(url, start_time, end_time, id, save=save)

        pass
        



            