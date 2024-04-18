import sys
import os
import subprocess
from FaceCropper.videoFaceCropper import VideoFaceCropper
from SamplesHandler import SamplesHandler

#sh = SamplesHandler()

#print(sh.get_text('s0\\wav\\s0-236-of-385.wav'))

samples_dir = 'test_parallel/raw'

samples_dir = sys.argv[1]

prepared_samples_dir = 'test_parallel/samples/'
samples_audio_dir = 'test_parallel/wav/'
alignments_dir = 'test_parallel/alignments/'

if len(sys.argv) > 2:
    prepared_samples_dir = sys.argv[2]
    samples_audio_dir = sys.argv[3]
    alignments_dir = sys.argv[4]

sh = SamplesHandler()

bad_samples = sh.handle_directory(directory=samples_dir, samples_directory=prepared_samples_dir, alignments_directory=alignments_dir, wav_directory=samples_audio_dir)

print('Bad samples:', bad_samples)
