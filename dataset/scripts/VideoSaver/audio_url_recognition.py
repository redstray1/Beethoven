from videoSaverModule import VideoSaver
import sys

url = sys.argv[1]

opts = {
        "writesubtitles" : True,
        "writeautomaticsub" : True, 
        "simulate" : True,
        "quiet" : True,
        "forceurl" : True,
        "format" : f"ba/b"
        }

vs = VideoSaver(opts)
vs.get_audio_url(url)

