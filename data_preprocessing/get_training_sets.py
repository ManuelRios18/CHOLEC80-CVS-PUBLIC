import glob
import pandas as pd


def get_stats():
    labels = glob.glob("../data/surgeons_annotations/*.csv")
    result = pd.DataFrame(columns=["video_name", "two_structures_score_1", "two_structures_score_2",
                                   "cystic_plate_score_1", "cystic_plate_score_2", "hc_triangle_score_1",
                                   "hc_triangle_score_2"])
    for l in labels:
        video_labels = pd.read_csv(l)
        result.loc[len(result)] = [video_labels.iloc[0]["video_name"],
                                   video_labels["two_structures_score"].value_counts().to_dict().get(1, 0),
                                   video_labels["two_structures_score"].value_counts().to_dict().get(2, 0),
                                   video_labels["cystic_plate_score"].value_counts().to_dict().get(1, 0),
                                   video_labels["cystic_plate_score"].value_counts().to_dict().get(2, 0),
                                   video_labels["hc_triangle_score"].value_counts().to_dict().get(1, 0),
                                   video_labels["hc_triangle_score"].value_counts().to_dict().get(2, 0)]

    return result


def get_set(video_list):
    result = list()
    for video_name in video_list:
        try:
            result.append(pd.read_csv(f"../data/surgeons_annotations/{video_name}"))
        except:
            print(f"Annotations for {video_name} does not exist")

    return pd.concat(result)


test_videos = ["video05.csv", "video07.csv", "video06.csv", "video09.csv", "video14.csv",
               "video24.csv", "video22.csv", "video26.csv", "video29.csv", "video33.csv",
               "video35.csv", "video39.csv", "video53.csv", "video55.csv", "video61.csv"]

val_videos = ["video01.csv", "video02.csv", "video10.csv", "video16.csv", "video17.csv",
              "video28.csv", "video32.csv", "video46.csv", "video47.csv", "video57.csv",
              "video59.csv", "video70.csv", "video63.csv", "video65.csv", "video67.csv"]

train_videos = [f"video{str(i).zfill(2)}.csv"
                for i in range(1, 81) if f"video{str(i).zfill(2)}.csv" not in test_videos + val_videos]

train = get_set(train_videos)
val = get_set(val_videos)
test = get_set(test_videos)
stats = get_stats()

train.to_csv("../data/train.csv", index=False)
val.to_csv("../data/val.csv", index=False)
test.to_csv("../data/test.csv", index=False)
stats.to_csv("../data/stats.csv", index=False)
