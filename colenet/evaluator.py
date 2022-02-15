import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Cholec80CSVEvaluator:

    def __init__(self, set_name, log_path, target_metric):
        self.log_file_path = os.path.join(log_path, f"{set_name}_log.csv")
        self.target_metric = target_metric
        self.predictions_list = list()
        self.labels_list = list()
        self.losses_list = list()
        self.best_score = -100
        self.log_headers = ["two_structures_accuracy", "cystic_plate_accuracy", "hc_triangle_score_accuracy",
                            "mean_acc", "two_structures_precision", "cystic_plate_precision",
                            "hc_triangle_score_precision", "mean_precision", "two_structures_recall",
                            "cystic_plate_recall", "hc_triangle_score_recall", "mean_recall", "two_structures_f1",
                            "cystic_plate_f1", "hc_triangle_score_f1", "mean_f1", "loss", "n_samples"]

        with open(self.log_file_path, 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def reset(self):
        self.predictions_list = list()
        self.labels_list = list()
        self.losses_list = list()

    def add_batch(self, predictions, labels, loss):
        labels = labels.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        self.predictions_list.append(predictions)
        self.labels_list.append(labels)
        self.losses_list.append(loss.item())

    def get_metrics(self):
        y = np.vstack(self.labels_list)
        y_hat = np.vstack(self.predictions_list)

        # Get Accuracy per criteria
        two_structures_accuracy = accuracy_score(y[:, 0], y_hat[:, 0])
        cystic_plate_accuracy = accuracy_score(y[:, 1], y_hat[:, 1])
        hc_triangle_score_accuracy = accuracy_score(y[:, 2], y_hat[:, 2])
        mean_acc = np.mean([two_structures_accuracy, cystic_plate_accuracy, hc_triangle_score_accuracy])

        # Get Precision per criteria
        two_structures_precision = precision_score(y[:, 0], y_hat[:, 0])
        cystic_plate_precision = precision_score(y[:, 1], y_hat[:, 1])
        hc_triangle_score_precision = precision_score(y[:, 2], y_hat[:, 2])
        mean_precision = np.mean([two_structures_precision, cystic_plate_precision, hc_triangle_score_precision])

        # Get Recall per criteria
        two_structures_recall = recall_score(y[:, 0], y_hat[:, 0])
        cystic_plate_recall = recall_score(y[:, 1], y_hat[:, 1])
        hc_triangle_score_recall = recall_score(y[:, 2], y_hat[:, 2])
        mean_recall = np.mean([two_structures_recall, cystic_plate_recall, hc_triangle_score_recall])

        # Get F1-Score per criteria
        two_structures_f1 = f1_score(y[:, 0], y_hat[:, 0])
        cystic_plate_f1 = f1_score(y[:, 1], y_hat[:, 1])
        hc_triangle_score_f1 = f1_score(y[:, 2], y_hat[:, 2])
        mean_f1 = np.mean([two_structures_f1, cystic_plate_f1, hc_triangle_score_f1])

        #Get epoch loss
        loss = np.sum(self.losses_list)

        new_row = pd.Series(data=[two_structures_accuracy, cystic_plate_accuracy, hc_triangle_score_accuracy, mean_acc,
                                  two_structures_precision, cystic_plate_precision, hc_triangle_score_precision,
                                  mean_precision, two_structures_recall, cystic_plate_recall, hc_triangle_score_recall,
                                  mean_recall, two_structures_f1, cystic_plate_f1, hc_triangle_score_f1, mean_f1,
                                  loss, y.shape[0]], index=self.log_headers)

        with open(self.log_file_path, 'a') as f:
            f.write(','.join(str(d) for d in new_row.values) + '\n')

        new_best_model = False
        if new_row[self.target_metric] > self.best_score:
            self.best_score = new_row[self.target_metric]
            new_best_model = True

        return new_best_model, self.best_score
