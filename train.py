from dataset import HAM10000, train_test
from model import SimpleNet, initialize_weights, Adversary

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import random

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def train_model():
    # establish random seed, ensure deterministic behavior throughout training
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.set_printoptions(linewidth=300)
    generator = torch.Generator().manual_seed(seed)

    # dataset creation
    path = "HAM10000/archive/"

    dataset = train_test(path)
    datasets_dict = dataset.getSplit(seed=seed)
    train = datasets_dict["train"]
    test = datasets_dict["test"]

    # model creation
    model = SimpleNet(3, 7)
    model.apply(initialize_weights)
    classes = {'nv': 0, 'bkl': 1, 'akiec': 2, 'df': 3, 'mel': 4, 'bcc': 5, 'vasc': 6}

    adversary = Adversary()
    model.apply(initialize_weights)

    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    criterion_adv = torch.nn.CrossEntropyLoss()
    optimizer_adv = torch.optim.SGD(adversary.parameters(), lr=0.001)

    adv_weight = 0.5

    # training parameters
    epochs = 100
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, drop_last=True, generator=generator, worker_init_fn=worker_init_fn)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    adversary.to(device)

    # TensorBoard writer
    writer = SummaryWriter()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        total = 0
        correct = 0
        loss = 0
        for i, (images, labels) in enumerate(train_loader):
            # data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # model application
            outputs = model(images)
            
            # loss formulation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            max_indices = torch.argmax(outputs, 1)
            one_hots = torch.zeros_like(outputs).scatter_(1, max_indices.unsqueeze(1), 1.0)

            correct += (one_hots * labels).sum().item()
            total += labels.size(0)
            loss += loss.item()

            # Log loss and outputs to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

            if total >= 5000:
                accuracy = 100 * correct / total
                writer.add_scalar('Accuracy/train', accuracy, epoch * len(train_loader) + i)
                
                print(f"Epoch: {epoch}, Batch: {i+1}, Loss: {loss.item()}, Accuracy: {accuracy}")

                total = 0
                correct = 0
                loss = 0

        torch.save(model.state_dict(), f"checkpoints8/model1_{epoch}.pth")

    # Close the writer
    writer.close()

def test_model_epoch():
    pass

def test_model_sex(ckpt):
    # establish random seed, ensure deterministic behavior throughout training
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.set_printoptions(linewidth=300)
    generator = torch.Generator().manual_seed(seed)

    # obtain dataset
    path = "HAM10000/archive/"
    dataset = train_test(path)
    datasets_dict = dataset.getSplit(seed=seed)
    train = datasets_dict["train"]
    test = datasets_dict["test"]

    # obtain model
    model = SimpleNet(3, 7)
    model.load_state_dict(torch.load(ckpt))
    model.eval()


    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    # set up fairness metrics
    classes = {'nv': 0, 'bkl': 1, 'akiec': 2, 'df': 3, 'mel': 4, 'bcc': 5, 'vasc': 6}
    class_nums = {0: 'nv', 1: 'bkl', 2: 'akiec', 3: 'df', 4: 'mel', 5: 'bcc', 6: 'vasc'}

    total = 0
    correct_outputs = 0

    results = {}

    print()
    print()
    print(f"Testing {ckpt}")
    for i, item in enumerate(test):
        image, label, attributes = item
        label_num = torch.argmax(label).item()

        image, label = image.to(device), label.to(device)
        image = image.unsqueeze(0)
        output = model(image)
        correct = torch.argmax(output).item() == label_num

        total += 1
        if correct:
            correct_outputs += 1

        if label_num not in results.keys():
            results[label_num] = [{"correct": int(correct), "age": attributes.age, "sex" : attributes.sex}]
        else:
            results[label_num].append({"correct": int(correct), "age": attributes, "sex" : attributes.sex})
    
    print(f"Total: {total}, Correct: {correct_outputs}, Accuracy: {correct_outputs/total}")
    print()

    for sex in ["male", "female"]:

        sex_correct = 0
        sex_total = 0

        for key in results.keys():
            correct = 0
            total = 0
            for item in results[key]:
                if item["sex"] == sex:
                    correct += item["correct"]
                    total += 1
                    sex_correct += item["correct"]
                    sex_total += 1

            print(f"Sex: {sex}, Class: {class_nums[key]}, Correct: {correct}, Total: {total}, Accuracy: {correct/total}")
        
        print(f"Sex: {sex}, Correct: {sex_correct}, total: {sex_total}, Accuracy: {100*sex_correct/sex_total}")
        print()

if __name__ == '__main__':
    train_model()
    # for i in range(70, 100):
    #     path = f"checkpoints8/model1_{i}.pth"
    #     test_model_fairness(path)
