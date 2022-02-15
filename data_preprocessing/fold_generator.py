import json
import random
import numpy as np
import pandas as pd

stats = pd.read_csv("../data/stats.csv")
stats["has_cp"] = (stats.cystic_plate_score_1 + stats.cystic_plate_score_2) > 0
videos_with_plates = list(stats[stats.has_cp].video_name.values)
videos_with_plates.sort()
other_videos = list(set(list(stats.video_name)) - set(videos_with_plates))

validation_folds = list()

validation_folds.append([videos_with_plates[0]] + other_videos[:4])
videos_with_plates = videos_with_plates[1:]
other_videos = other_videos[4:]

for _ in range(15):
    validation_folds.append(videos_with_plates[:2] + other_videos[:3])
    videos_with_plates = videos_with_plates[2:]
    other_videos = other_videos[3:]

with open('../data/folds.json', 'w') as fp:
    json.dump({"folds": validation_folds}, fp)
