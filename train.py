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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion_adv = torch.nn.CrossEntropyLoss()
    optimizer_adv = torch.optim.Adam(adversary.parameters(), lr=0.001)

    adv_weight = 0.1

    # training parameters
    epochs = 100
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, drop_last=True, generator=generator, worker_init_fn=worker_init_fn)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    adversary.to(device)
    do_adversary = False
    do_mixup = False

    # TEMP
    # do_adversary = True
    do_mixup = True
    train_loader.dataset.set_do_mixup(False)

    # TensorBoard writer
    writer = SummaryWriter()

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        total = 0
        loss_progress = 0
        adv_loss_progress = 0
        final_loss = 0
        for i, (images, labels, sex) in enumerate(train_loader):
            # data
            if False:
                print(f"labels: {labels[0]}")
                print(f"sex: {sex[0]}")
                plt.imshow(images[0].permute(1, 2, 0))
                plt.show()
            images, labels, sex = images.to(device), labels.to(device), sex.to(device)
            optimizer.zero_grad()
            optimizer_adv.zero_grad()
            
            # model application
            outputs, embedding = model(images)
            embedding = embedding.detach()

            loss, adv_loss = None, None
            if i % 100 == 0:
                print(f"\nepoch: {epoch}, iter: {i}")
                print(f"outputs: {outputs[0]}")
                print(f"truth: {labels[0]}")
            
            if do_adversary:
                adv_outputs = adversary(embedding)
                adv_loss = criterion_adv(adv_outputs, sex)
                adv_loss.backward()
                optimizer_adv.step()

                loss = criterion(outputs, labels) - adv_weight * adv_loss.detach()
                loss.backward()
                optimizer.step()
            else:
                # loss formulation
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # max_indices = torch.argmax(outputs, 1)
            # one_hots = torch.zeros_like(outputs).scatter_(1, max_indices.unsqueeze(1), 1.0)

            total += 1
            loss_progress += loss.item()

            if do_adversary:
                adv_loss_progress += adv_loss.item()
                final_loss = loss.item() - adv_loss.item() * adv_weight

            # Log loss and outputs to TensorBoard
            if total >= 100:
                writer.add_scalar('Model Loss/train', loss.item()/total, epoch * len(train_loader) + i)
                
                if do_adversary:
                    writer.add_scalar('Adversary Loss/train', adv_loss.item()/total, epoch * len(train_loader) + i)
                    writer.add_scalar('Final Loss/train', final_loss/total, epoch * len(train_loader) + i)

                total = 0
                loss_progress = 0
                adv_loss_progress = 0

        torch.save(model.state_dict(), f"checkpoints_fixed_2/model_{epoch}.pth")
        torch.save(adversary.state_dict(), f"checkpoints_fixed_2/adversary_{epoch}.pth")

        # switch to interventions
        # if epoch == 5:
        #     print('switching sides')
        #     # do_adversary = True
        #     do_mixup = True
        #     train_loader.dataset.set_do_mixup(True)

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

    n = 2
    if n == 0:
        train_model()
    elif n == 1:
        for i in range(0, 101):
            path = f"checkpoints_fixed_1/model_{i}.pth"
            test_model_sex(path)
    elif n == 2:
        for i in range(0, 101):
            path = f"checkpoints_fixed_2/model_{i}.pth"
            test_model_sex(path)
    # for i in range(70, 100):
    #     path = f"checkpoints8/model1_{i}.pth"
    #     test_model_fairness(path)
    # for i in range(0, 30):
    #     path = f"checkpoints_fixed_1/model_{i}.pth"
    #     test_model_sex(path)
