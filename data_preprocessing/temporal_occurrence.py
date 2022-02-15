import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Script to check the when does the CVS occurs within a video.
Plot an histogram of it
"""

video_fps = 25
annotations = pd.read_excel("../data/surgeons_annotations.xlsx")
final_frames = pd.read_csv("../results/videos_frames_index.csv")
video_name = 80
criteria = ['two_structures', 'cystic_plate', 'hepatocystic_triangle']
criteria_pretty = {"two_structures": "Two Structures",
                   "cystic_plate": "Cystic Plate",
                   "hepatocystic_triangle": "Hepatocystic Triangle"}

criteria_occurrence = {criterion: {2: np.zeros(101), 1: np.zeros(101)} for criterion in criteria}

# For each video
for video_name in range(1, 81):
    # For each criterion
    for criterion in criteria:
        # Get annotations and final valid frame
        video_annotations = annotations[annotations.video == video_name]
        video_final_frame = int(final_frames[final_frames.video_name ==
                                             f"video{str(video_name).zfill(2)}"]["final_frame"])
        # for each possible score
        for score in [1, 2]:
            # Get the annotations of that score
            valid_annotations = video_annotations[video_annotations[criterion] == score]
            # For each of those annotations
            for i, row in valid_annotations.iterrows():
                # Extract initial and final frame
                initial_frame = int(row["initial_minute"] * 60 * video_fps + row["initial_second"] * video_fps)
                final_frame = int(row["final_minute"] * 60 * video_fps + row["final_second"] * video_fps)

                # Compute occurrence
                occurrence_rate = 100 * (np.array(list(range(initial_frame, final_frame))) / video_final_frame)
                bins = np.digitize(occurrence_rate, list(range(0, 101))) - 1
                inds, counts = np.unique(bins, return_counts=True)
                for ind, count, in zip(inds, counts):
                    criteria_occurrence[criterion][score][ind] += count

# Plot all curves
cut_off = 75
for criterion in criteria:
    plt.figure()
    plt.title(criteria_pretty[criterion])
    plt.plot(criteria_occurrence[criterion][1], label="Score 1", color="tomato")
    plt.plot(criteria_occurrence[criterion][2], label="Score 2", color="turquoise")
    plt.vlines(cut_off, 0, np.max(criteria_occurrence[criterion][1]), color="gold", label="Cut Off")
    plt.legend()
    plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
    plt.ylabel("N. Frames")
    plt.xlabel("Normalized occurrence")
    plt.show()
    plt.savefig(f"../results/{criterion}.png")


plt.figure()
plt.title("Occurrence Rate")
plt.plot(criteria_occurrence["two_structures"][1], '-', label="Two Structures Score 1", color="tomato")
plt.plot(criteria_occurrence["two_structures"][2], '--', label="Two Structures Score 2", color="tomato")
plt.plot(criteria_occurrence["cystic_plate"][1], '-', label="Cystic Plate Score 1", color="turquoise")
plt.plot(criteria_occurrence["cystic_plate"][2], '--', label="Cystic Plate Score 2", color="turquoise")
plt.plot(criteria_occurrence["hepatocystic_triangle"][1], '-', label="H. Triangle Score 1", color="gold")
plt.plot(criteria_occurrence["hepatocystic_triangle"][2], '--', label="H. Triangle Score 2", color="gold")
plt.vlines(cut_off, 0, np.max(criteria_occurrence["hepatocystic_triangle"][1]), color="black",
           linewidth=3, label="Cut Off")
plt.legend()
plt.grid(color='black', linestyle='--', linewidth=1, alpha=0.15)
plt.ylabel("N. Frames")
plt.xlabel("Normalized occurrence")
plt.show()
plt.savefig(f"../results/total.png")