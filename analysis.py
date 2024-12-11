import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class skintone_analysis:
    def __init__(self, dataset):
        self.classes = {'nv': 0, 'bkl': 1, 'akiec': 2, 'df': 3, 'mel': 4, 'bcc': 5, 'vasc': 6}
        self.dataset = dataset

    def analyze_dataset(self):
        '''
        Analyze the dataset to determine the distribution of skin tones
        '''
        for i, (image, label) in enumerate(self.dataset):
            print(label)
            
if __name__ == "__main__":
    # # establish random seed
    # seed = 42
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    vals1 = [0.0, 0.1, 0.4, 0.9, 1.6]
    vals2 = [0.0, 1, 2, 3, 4]

    plt.plot(vals2, vals1, marker='o')
    plt.title("Test Plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid()
    plt.xticks(vals1)
    plt.show()