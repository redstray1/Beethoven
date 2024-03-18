from videoSaverModule import VideoSaver
import sys

url = sys.argv[1]

start = '00:00:00'
end = '00:00:15'

start = sys.argv[2]
end = sys.argv[3]

target_path = 'downloaded_trimmed_video.mp4'

if len(sys.argv) >= 5:
    target_path = sys.argv[4]

quality = 720

if len(sys.argv) >= 6:
    quality = int(sys.argv[5])
opts = {
        "writesubtitles" : True,
        "writeautomaticsub" : True,
        "simulate" : True,
        "quiet" : True,
        "forceurl" : True,
        "format" : f"wv*[ext=mp4][height={quality}]+ba/b"
        }

vs = VideoSaver(opts)

vs.download(url, start, end, output_path=target_path)

