import os
import math
import numpy as np
import pandas as pd


annotations_file_name = "surgeons_annotations"
truncation_ratio = 0.85
target_fps = 5


video_fps = 25
doctors_annotations = pd.read_excel(f"../data/{annotations_file_name}.xlsx", engine='openpyxl')
result_path = f"../data/{annotations_file_name.split('.')[0]}"
index_data = pd.read_csv("../results/videos_frames_index.csv")

for video_tag, video_data in doctors_annotations.groupby("video"):
    print(f"Processing video {video_tag}")
    video_name = f"video{str(video_tag).zfill(2)}"
    valid_frames = index_data[index_data["video_name"] == video_name]["final_frame"].iloc[0]

    video_result = pd.DataFrame(columns=["video_name", "image", "two_structures_score", "cystic_plate_score",
                                         "hc_triangle_score"])
    video_result["image"] = list(range(valid_frames + 1))
    video_result[["video_name", "two_structures_score", "cystic_plate_score",
                  "hc_triangle_score"]] = [video_name, 0, 0, 0]

    for i, row in video_data.iterrows():
        if row["two_structures"] + row["cystic_plate"] + row["hepatocystic_triangle"] > 0:
            tag = [row["two_structures"], row["cystic_plate"], row["hepatocystic_triangle"]]
            initial_frame = int(row["initial_minute"] * 60 * video_fps + row["initial_second"] * video_fps)
            final_frame = int(row["final_minute"] * 60 * video_fps + row["final_second"] * video_fps)
            assert initial_frame <= video_result.shape[0], "ERROR: initial frame is out of range"
            if final_frame >= video_result.shape[0]:
                final_frame = valid_frames
            for frame_number in range(initial_frame, final_frame+1):
                video_result.loc[frame_number, ["two_structures_score",
                                                "cystic_plate_score", "hc_triangle_score"]] = tag

    # truncate videos so we get rid off of a bunch of zeros

    initial_second = math.floor(video_result.shape[0] * truncation_ratio / 25)
    initial_frame = initial_second * 25
    video_result = video_result[initial_frame:]

    # Reduce FPS by randomly extracting frames

    chunk_list = list()
    for offset in range(0, video_result.shape[0], video_fps):
        chunk = video_result[offset:offset+video_fps]
        chunk.reset_index(inplace=True, drop=True)
        replace_flag = False
        if chunk.shape[0] < target_fps:
            replace_flag = True
        index = list(np.random.choice(chunk.shape[0], target_fps, replace=replace_flag))
        chunk = chunk.iloc[index]
        chunk_list.append(chunk)

    video_result = pd.concat(chunk_list)
    video_result.to_csv(os.path.join(result_path, f"{video_name}.csv"), index=False)