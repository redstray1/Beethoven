import sys
import os
import subprocess
import speech_recognition as s_r
from FaceCropper.videoFaceCropper import VideoFaceCropper

class SamplesHandler():

    def __init__(self):
        self.vfc = VideoFaceCropper()
        self.recognizer = s_r.Recognizer()

    def get_text(self, audiofile):
        #audio_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), audiofile)
        audio_path = audiofile
        sample = s_r.AudioFile(audio_path)
        print(sample.DURATION)

        with sample as audio:
            
            #self.recognizer.adjust_for_ambient_noise(audio)
            content = self.recognizer.record(audio)
            return self.recognizer.recognize_google(content, language='ru-RU')
    
    def extract_audio(self, videofile, *, output_directory, acodec='pcm_s16le'):
        filebase = ".".join(videofile.split(".")[:-1])
        filename = '.'.join(os.path.basename(videofile).split('.')[:-1])

        output_path = os.path.join(output_directory, filename)

        fileext = videofile.split(".")[-1]
        extract_cmd = ["ffmpeg", "-y", "-i", videofile, "-vn",
                     "-acodec", acodec, "-ar", '44100', "-ac", '2', output_path + '.wav']
        subprocess.check_output(extract_cmd)
        return output_path + '.wav'

    def handle_directory(self, *, directory='data/', samples_directory='data/samples', alignments_directory='data/alignments', wav_directory='data/wav', audio_ext = '.mp4'):
        samples = []
        
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), directory)
        samples_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), samples_directory)
        alignments_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), alignments_directory)
        wav_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), wav_directory)

        print(os.listdir(directory))
        
        for file in os.listdir(directory):
            if file.endswith(audio_ext):
                samples.append((os.path.join(directory, file), file))
                print(file)
        bad_samples = 0

        os.makedirs(samples_directory, exist_ok=True)
        os.makedirs(alignments_directory, exist_ok=True)
        os.makedirs(wav_directory, exist_ok=True)

        for sample, filename in samples:
            success = self.vfc.crop_video(sample, os.path.join(samples_directory, filename), bad_frames_threshold=30)
            if not success:
                bad_samples += 1
            else:
                filebase = ".".join(sample.split(".")[:-1])
                result_path = self.extract_audio(sample, output_directory=wav_directory)
                text = self.get_text(result_path)


                alignment_name = '.'.join(filename.split('.')[:-1])
                with open(os.path.join(alignments_directory, alignment_name + '.txt'), 'w') as align:
                    align.write(text)
                    #ВОТ ТУТ ПРОБЛЕМА
        return bad_samples
