import sys
from beetLibrary import create_format_opts
from dataCollector import DataCollector

data = sys.argv[1]

opts = create_format_opts(resolution=720)


dc = DataCollector(opts)

dc.collect_samples(data, save=True)