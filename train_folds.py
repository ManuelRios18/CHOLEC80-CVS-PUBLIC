import json
from utils import get_set
from colenet.trainer import ColenetTrainer


def get_train_val(fold_id):
    with open('data/folds.json') as json_file:
        all_folds = json.load(json_file)
    val_videos = list(all_folds["folds"][fold_id])
    all_videos = [f"video{str(i).zfill(2)}" for i in range(1, 81)]
    train_videos = list(set(all_videos) - set(val_videos))

    return get_set(train_videos), get_set(val_videos)


def get_stats(data_df, verbose=True):
    pos_weight = list()
    for criterion in ["two_structures_score", "cystic_plate_score", "hc_triangle_score"]:
        n_neg = data_df[criterion].value_counts()[0]
        n_pos = (data_df.shape[0]-n_neg)
        weight = n_neg/n_pos
        pos_weight.append(weight)
        if verbose:
            print(f"Criterion {criterion} - N. Positive: {n_pos} - N. negative: {n_neg} - Weight: {weight}")

    return pos_weight


root_dir = "/media/manuel/DATA/datasets/COLELAPS FRAMES"
backbone = "vgg"

epochs = 10
batch_size = 32
learning_rate = 1e-5

for i in range(16):
    print("Starting training for fold ", i)
    train, val = get_train_val(i)
    print("train")
    pos_weight = get_stats(train)
    print("val")
    get_stats(val)
    log_name = f"{backbone}_fold_{i}"
    trainer = ColenetTrainer(root_dir, backbone, log_name, "mean_f1", pos_weight, train_set=train, val_set=val)
    trainer.train_colenet(epochs, batch_size, learning_rate)
