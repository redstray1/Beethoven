import yt_dlp
import sys
import ffmpeg
import json

def trim(input_path, output_path, start=30, end=60):
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

def convert_time_to_secs(time):
    lst = time.split(':')
    pw = 1
    cur = 0
    for x in lst[::-1]:
        cur += pw * int(x)
        pw *= 60
    return cur

url = sys.argv[1]

start = 0.0
end = 15.0

if len(sys.argv) >= 3:
    start = convert_time_to_secs(sys.argv[2])
if len(sys.argv) >= 4:
    end = convert_time_to_secs(sys.argv[3])

target_path = 'downloaded_trimmed_video.mp4'

if len(sys.argv) >= 5:
    target_path = sys.argv[4]

quality = 720

if len(sys.argv) >= 6:
    quality = int(sys.argv[5])
opts = {
        "writesubtitles" : False,
        "writeautomaticsub" : False,
        "simulate" : True,
        "quiet" : True,
        "forceurl" : True,
        "format" : f"wv*[ext=mp4][height={quality}]+ba/b"
        }

with yt_dlp.YoutubeDL(opts) as ydl:
    info = ydl.extract_info(url, download=False)

    full_url = info['url']

    trim(full_url, 'test.mp4', start, end)

  # print(info['url'])

