import os
import tqdm
import torch
from torchvision import transforms
from colenet.colenet_model import ColeNet
from colenet.evaluator import Cholec80CSVEvaluator
from colenet.cholec80csv_dataset import Cholec80CSVDataset

backbone = "vgg"
model_path = f"log/{backbone}/best_model.pth"
root_dir = "/media/manuel/DATA/datasets/COLELAPS FRAMES"

print("Loading Model")
model = ColeNet(backbone)
model.load_state_dict(torch.load(model_path))
model.eval()

data_normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
test_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224),
                                      transforms.ToTensor(), data_normalization])

test_dataset = Cholec80CSVDataset(set_name="test", root_dir=root_dir, transform=test_transforms)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
evaluator = Cholec80CSVEvaluator("test", '/'.join(model_path.split('/')[:-1]), "mean_f1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
model.eval()
model.to(device)
evaluator.reset()
for sample in tqdm.tqdm(loader, total=len(loader)):
    inputs, labels, _ = sample
    inputs, labels = inputs.to(device), labels.to(device)
    logits = model(inputs.float())
    labels = labels.type_as(logits)
    outputs = torch.sigmoid(logits)
    evaluator.add_batch(outputs, labels, torch.tensor(0))
evaluator.get_metrics()
