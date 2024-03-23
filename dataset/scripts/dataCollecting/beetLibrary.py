import pandas as pd

def create_format_opts(resolution : int = 720):
    opts = {
        "writesubtitles" : False,
        "writeautomaticsub" : False, 
        "simulate" : True,
        "quiet" : True,
        "forceurl" : True,
        "format" : f"bv*[ext=mp4][height={resolution}]+ba/b"
        }
    return opts

def convert_time_to_secs(time : str):
        lst = time.split(':')
        pw = 1
        cur = 0
        for x in lst[::-1]:
            cur += pw * int(x)
            pw *= 60
        return cur

def read_table(data):
        if '.csv' in data:
            return pd.read_csv(data, dtype=str)
        if '.xlsx' in data:
            return pd.read_excel(data, dtype=str)