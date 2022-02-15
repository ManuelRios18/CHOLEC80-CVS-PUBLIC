import os
import utils
import pandas as pd

"""
DESCRIPTION:

This script extracts the index of the last valid frame, which corresponds with the last frame 
before the Clipping and Cutting Phase.
"""

dataset_path = utils.get_dataset_path()
dataset_path = os.path.join(dataset_path, "phase_annotations")
annotations_path = os.listdir(dataset_path)
result = pd.DataFrame(columns=["video_name", "final_frame"])
for annotation in annotations_path:
    print(f"Proccesing video {annotation}")
    data = pd.read_csv(os.path.join(dataset_path, annotation), sep='\t')
    frame_index = data[data["Phase"] == "ClippingCutting"]["Frame"].iloc[0] - 1
    result.loc[len(result)] = [annotation.split('-')[0], frame_index]

result.to_csv("../results/videos_frames_index.csv", index=False)