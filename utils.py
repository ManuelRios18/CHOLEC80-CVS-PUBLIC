import json
import pandas as pd


def get_dataset_path():
    f = open('../config/config.json')
    data = json.load(f)

    return data["dataset_path"]


def get_config():
    f = open('../config/config.json')
    data = json.load(f)

    return data


def get_set(video_list):
    result = list()
    for video_name in video_list:
        try:
            result.append(pd.read_csv(f"data/surgeons_annotations/{video_name}.csv"))
        except:
            print(f"Annotations for {video_name} does not exist")

    return pd.concat(result)
