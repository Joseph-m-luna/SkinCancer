import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleNet

class PAD_dataset(nn.Module):
    def __init__(self, path="PAD-UFES-20/dataset/"):
        self.classes = {'nv': 0, 'bkl': 1, 'akiec': 2, 'df': 3, 'mel': 4, 'bcc': 5, 'vasc': 6}
        # NV: Melanocytic nevi/NA
        # BKL: Benign keratosis-like lesions/NA
        # AKIEC: Actinic keratoses and intraepithelial carcinoma / Bowen's disease/ACK
        # DF: Dermatofibroma/NA
        # Mel: Melanoma/MEL
        # BCC: Basal cell carcinoma/BCC
        # VASC: Vascular lesions

        self.path = Path(path)
        self.pd = pd.read_csv(self.path / "metadata.csv")
        print(self.pd["fitspatrick"].value_counts())
        self.analyze_dataset()

    def analyze_dataset(self):
        '''
        Analyze the dataset to determine the distribution of skin tones
        '''
        # print(self.pd.head())
        print(self.pd[self.pd["fitspatrick"] == 6])

        common_tags = self.pd[self.pd["diagnostic"].str.contains("MEL|BCC", case=False, na=False)]
        print("counts", common_tags["fitspatrick"].value_counts())
        self.pd = common_tags

    def __len__(self):
        return len(self.pd)
    
    def __getitem__(self, idx):
        row = self.pd.iloc[idx]
        filename = row["img_id"]
        filepath = self.path / "combined" / filename

        image = Image.open(filepath)

        shape = np.array(image).shape
        height = (450 * shape[0])/600
        image = image.crop((0, (shape[0] -  height)/2,shape[0], ((shape[0] - height)/2) + height))
        image = image.resize((600, 450))
        image = image.convert("RGB")

        diagnostic = row["diagnostic"]
        result = -1
        if diagnostic == "BCC":
            result = 5
        elif diagnostic == "MEL":
            result = 4

        image = torch.from_numpy(np.array(image).transpose(2, 0, 1))/255

        gt = torch.zeros(7)
        gt[result] = 1.0
        
        return image, gt, row["fitspatrick"]
    
def test():
    dataset = PAD_dataset()
    model = SimpleNet(3, 7)
    name = "model1_98"
    #high performance: 98, 92, 90
    print(f"Loading model: {name}")
    model.load_state_dict(torch.load(f"checkpoints7/{name}.pth"))
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    data_tracker = {}
    data_tracker["correct"] = []
    data_tracker["total"] = 0
    data_tracker["incorrect"] = []
    for i, (image, gt, fitspatrick) in enumerate(dataset):
        image, gt, fitspatrick = image.to(device), gt.to(device), fitspatrick

        image = image.unsqueeze(0)

        output = model(image)

        max_val, max_index = torch.max(output, dim=1)
        if gt[int(max_index)] == 1:
            data_tracker["correct"].append(i)
        else:
            data_tracker["incorrect"].append(i)
    print(f"Accuracy: {len(data_tracker['correct'])/len(dataset)}, total: {len(dataset)}")
    
    incorrect = {}
    correct = {}
    for i in data_tracker["correct"]:
        dataset.pd.iloc[i]["fitspatrick"]
        if dataset.pd.iloc[i]["fitspatrick"] in correct:
            correct[dataset.pd.iloc[i]["fitspatrick"]] += 1
        else:
            correct[dataset.pd.iloc[i]["fitspatrick"]] = 1
    
    for i in data_tracker["incorrect"]:
        dataset.pd.iloc[i]["fitspatrick"]
        if dataset.pd.iloc[i]["fitspatrick"] in incorrect:
            incorrect[dataset.pd.iloc[i]["fitspatrick"]] += 1
        else:
            incorrect[dataset.pd.iloc[i]["fitspatrick"]] = 1
    
    print("correct", correct)
    print("incorrect", incorrect)
    for x in range(1, 6):
        print("Fitspatrick", x)
        if x in correct.keys():
            print("\tCorrect", correct[x])
        else:
            print("\tCorrect", 0)
        if x in incorrect.keys():
            print("\tIncorrect", incorrect[x])
        else:
            print("\tIncorrect", 0)

if __name__ == "__main__":
    test()
    # dataset = PAD_dataset()
    # dataset.analyze_dataset()

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")

    # for i, (image, label) in enumerate(dataset):
    #     print("new instance")
    #     time.sleep(1)
    #     # image, label = image.to(device), label.to(device)