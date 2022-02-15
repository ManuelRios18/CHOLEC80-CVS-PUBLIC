import os
import copy
import tqdm
import torch
import shutil
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from colenet.colenet_model import ColeNet
from colenet.evaluator import Cholec80CSVEvaluator
from colenet.cholec80csv_dataset import Cholec80CSVDataset


class ColenetTrainer:

    def __init__(self, root_dir, backbone, log_name, target_metric, pos_weight=None, train_set=None, val_set=None):
        seed = 990411
        self.root_dir = root_dir
        self.backbone = backbone
        self.target_metric = target_metric
        self.set_random_seed(seed=seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device", self.device)
        data_normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256),
                                                     transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(), data_normalization])
        val_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256),
                                             transforms.CenterCrop(224), transforms.ToTensor(), data_normalization])
        self.train_dataset = Cholec80CSVDataset(set_name="train", root_dir=self.root_dir, transform=train_transforms,
                                                predefined_set=train_set)
        self.val_dataset = Cholec80CSVDataset(set_name="val", root_dir=self.root_dir, transform=val_transforms,
                                              predefined_set=val_set)
        self.log_path = self.create_log_dir(log_name)
        self.best_model = None
        # Pos Weight as #negative/#positive per class
        if pos_weight is None:
            self.pos_weight = torch.Tensor([1.4380261927034612, 6.650975047179703,
                                            2.993378570646821]).to(device=self.device)
        else:
            self.pos_weight = torch.Tensor(pos_weight).to(device=self.device)
        print("Using pos_weight", self.pos_weight)

    def set_random_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def create_log_dir(self, log_name):
        logs_dir = os.path.join("log", log_name)
        if os.path.isdir(logs_dir):
            shutil.rmtree(logs_dir)
        os.mkdir(logs_dir)

        return logs_dir

    def save_model(self, model, model_name):
        with open(f"{self.log_path}/{model_name}.pth", 'wb') as fp:
            state = model.state_dict()
            torch.save(state, fp)

    def train_epoch(self, model, optimizer, criterion, epoch, train_loader, evaluator):
        model.train()
        evaluator.reset()
        for sample in tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch}"):
            inputs, labels, _ = sample
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            logits = model(inputs.float())
            labels = labels.type_as(logits)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            outputs = torch.sigmoid(logits)
            evaluator.add_batch(outputs, labels, loss)
        evaluator.get_metrics()

    def validate_epoch(self, model, optimizer, criterion, epoch, val_loader, evaluator):
        model.eval()
        evaluator.reset()
        for sample in tqdm.tqdm(val_loader, total=len(val_loader), desc=f"Val epoch {epoch}"):
            inputs, labels, _ = sample
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = model(inputs.float())
            labels = labels.type_as(logits)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)
            evaluator.add_batch(outputs, labels, loss)
        new_best, new_score = evaluator.get_metrics()
        if new_best:
            print(f"New best {self.target_metric}", new_score)
            self.best_model = copy.deepcopy(model)
            self.save_model(self.best_model, "best_model")

    def train_colenet(self, epochs, batch_size, learning_rate):
        print(f"Starting training with {self.backbone} backbone")
        model = ColeNet(backbone=self.backbone)
        model.to(self.device)
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        train_evaluator = Cholec80CSVEvaluator("train", self.log_path, self.target_metric)
        val_evaluator = Cholec80CSVEvaluator("val", self.log_path, self.target_metric)
        for epoch in range(epochs):
            print(f"epoch {epoch + 1} of {epochs}")
            self.train_epoch(model, optimizer, criterion, epoch, train_loader, train_evaluator)
            self.validate_epoch(model, optimizer, criterion, epoch, val_loader, val_evaluator)

        return "yes"

