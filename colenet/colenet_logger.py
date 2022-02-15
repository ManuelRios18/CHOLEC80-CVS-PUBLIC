import os


class Logger:

    def __init__(self, log_name):
        self.log_path = os.path.join("logs", f"{log_name}.csv")
        self.log_headers = [
            'epoch',
            'train-loss',
            'train-two_structures_score-precision',
            'train-two_structures_score-recall',
            'train-two_structures_score-f1',
            'train-cystic_plate_score-precision',
            'train-cystic_plate_score-recall',
            'train-cystic_plate_score-f1',
            'train-hc_triangle_score-precision',
            'train-hc_triangle_score-recall',
            'train-hc_triangle_score-f1',
            'train-harmonic-f1',
            'test-loss',
            'test-two_structures_score-precision',
            'test-two_structures_score-recall',
            'test-two_structures_score-f1',
            'test-cystic_plate_score-precision',
            'test-cystic_plate_score-recall',
            'test-cystic_plate_score-f1',
            'test-hc_triangle_score-precision',
            'test-hc_triangle_score-recall',
            'test-hc_triangle_score-f1',
            'test-harmonic-f1'
        ]
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        with open(self.log_path, 'w') as f:
            f.write(','.join(self.log_headers) + '\n')

    def save_row(self, data):
        with open(self.log_path, 'a') as f:
            f.write(','.join(str(d) for d in data) + '\n')