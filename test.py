from model import SimpleNet
import torch
from dataset import train_test
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



def test_model():
     # test model
    model = SimpleNet(3, 7)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    path = "HAM10000/archive/"
    dataset = train_test(path)
    datasets_dict = dataset.getSplit()

    test_type = "test"
    
    if test_type == "test":
        test = datasets_dict["test"]
        file = "accuracy_over_time.npy"
    elif test_type == "train":
        test = datasets_dict["train"]
        file = "accuracy_over_time_train_set.npy"

    if os.path.exists(file):
        accuracy_over_time = list(np.load(file))
        start = len(accuracy_over_time)
    else:
        accuracy_over_time = []
        start = 0

    end = 35
    for i in range(start, end):
        # zero data tracker
        data_tracker = {}
        data_tracker["correct"] = []
        data_tracker["total"] = 0
        data_tracker["incorrect"] = []

        # initialize model
        model.load_state_dict(torch.load(f"checkpoints8/model1_{i}.pth"))
        model.eval()
        model.to(device)

        # test model
        correct_pigments = 0
        pigmentation_threshold = 200
        for i, (image, label, pigment) in tqdm(enumerate(test), total=len(test)):
            data_tracker["total"] += 1
            image, label = image.to(device), label.to(device)

            image = image.unsqueeze(0)
            
            output = model(image)
            
            max_val, max_index = torch.max(output, dim=1)
            if label[int(max_index)] == 1:
                data_tracker["correct"].append(i)
            else:
                data_tracker["incorrect"].append(i)
    
        print(f"Accuracy: {len(data_tracker['correct'])/data_tracker['total']}, total: {data_tracker['total']}")
        accuracy_over_time.append(len(data_tracker['correct'])/data_tracker['total'])
    np.save(file, accuracy_over_time)
    
    #visualize accuracy over time
    plt.plot(np.arange(len(accuracy_over_time)), accuracy_over_time, marker='o')
    plt.title("Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    print(np.arange(len(accuracy_over_time)))
    print(accuracy_over_time)
    # plt.xticks(np.arange(len(accuracy_over_time)))
    plt.show()

if __name__ == "__main__":
    test_model()