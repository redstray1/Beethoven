import sys
import os
import subprocess
from FaceCropper.videoFaceCropper import VideoFaceCropper
from SamplesHandler import SamplesHandler

samples_dir = 'data/'

samples_dir = sys.argv[1]

prepared_samples_dir = 'demo_data/samples/'
samples_audio_dir = 'demo_data/wav/'
alignments_dir = 'demo_data/alignments/'

sh = SamplesHandler()

bad_samples = sh.handle_directory(directory=samples_dir, samples_directory=prepared_samples_dir, alignments_directory=alignments_dir, wav_directory=samples_audio_dir)

print('Bad samples:', bad_samples)
