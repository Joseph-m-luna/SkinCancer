import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt


class HAM10000(Dataset):
    def __init__(self, df, path, equalize=False, augment=True, testing=False):
        '''
        Initialize dataset with HAM10000 data based on Kaggle dataset structure

        Parameters:
        path (str): path to archive folder containing CSVs and data (ex. "HAM10000/archive/")
        '''

        self.path = Path(path)
        self.df = df
        self.classes = {'nv': 0, 'bkl': 1, 'akiec': 2, 'df': 3, 'mel': 4, 'bcc': 5, 'vasc': 6}
        self.seed=42
        self.do_augment = augment
        if testing:
            pass
            # self.collect_skin_data()
        self.testing = testing
        if equalize:
            self.balance()

    def __len__(self):
        # Return length of dataset
        return len(self.df)
    
    def collect_skin_data(self):
        '''
        Collect skin data from HAM10000 dataset

        Returns:
        data (DataFrame): DataFrame containing skin data
        '''
        if os.path.exists(self.path / "combined" / "pigmentatino.pkl"):
            pass # read existing data if present
        else:
            # generate pigmentation data
            self.pigment_dict = {}

            for i in range(len(self)):
                row = self.df.iloc[i]
                filepath = self.path / "combined" / (row["image_id"] + ".jpg")
                image = Image.open(filepath)

                # get pigmentation data using same methodology as used in https://onlinelibrary.wiley.com/doi/full/10.1002/jvc2.477#:~:text=However%2C%20the%20HAM10000%20data%20set,associated%20with%20lighter%20skin%20tones.
                # L_crop = (48, 36, 48 + 24, 36 + 18)
                # R_crop = (528, 36, 528 + 24, 36 + 18)
                # T_crop = (48, 396, 48 + 24, 396 + 18)
                # B_crop = (528, 396, 528 + 24, 396 + 18)
                
                L_crop = (48, 36, 48 + 24, 36 + 18)
                R_crop = (528, 36, 528 + 24, 36 + 18)
                T_crop = (48, 396, 48 + 24, 396 + 18)
                B_crop = (528, 396, 528 + 24, 396 + 18)
                


                TL_np = np.array(image.crop(L_crop))
                TR_np = np.array(image.crop(R_crop))
                BL_np = np.array(image.crop(T_crop))
                BR_np = np.array(image.crop(B_crop))

                image_draw = ImageDraw.Draw(image)
                image_draw.rectangle(L_crop, outline="red", width=3)
                image_draw.rectangle(R_crop, outline="red", width=3)
                image_draw.rectangle(T_crop, outline="red", width=3)
                image_draw.rectangle(B_crop, outline="red", width=3)

                red = []
                green = []
                blue = []
                for j in [TL_np, TR_np, BL_np, BR_np]:
                    red.append(np.median(j[:,:,0]))
                    green.append(np.median(j[:,:,1]))
                    blue.append(np.median(j[:,:,2]))

                self.pigment_dict[row["image_id"]] = {"RGB": {"red": np.mean(red), "green": np.mean(green), "blue": np.mean(blue)}, "mean": np.mean([np.mean(red), np.mean(green), np.mean(blue)])}
                if i % 100 == 0:
                    print(f"Processed {i} images")
    
    def get_pigmentation(self, index):
        '''
        Get pigmentation data for image

        Parameters:
        image_id (str): image ID to get pigmentation data for

        Returns:
        pigment_data (dict): pigmentation data for image
        ''' 
        return self.pigment_dict[self.df.iloc[index]["image_id"]]
    
    def check_threshold(self, threshold=100):
        '''
        Check if pigmentation data is above threshold

        Parameters:
        threshold (int): threshold to check pigmentation data against

        Returns:
        above_threshold (int): number of images with pigmentation data above threshold
        '''
        above_threshold = 0
        keys = []
        for key in self.pigment_dict:
            print(self.pigment_dict[key])
            if self.pigment_dict[key]["mean"] < threshold and self.pigment_dict[key]["mean"] > 40:
                keys.append(key)
                above_threshold += 1
        print("below threshold:", above_threshold)
        for key in keys:
            print(self.pigment_dict[key]["mean"])
            # plt.imshow(Image.open(self.path / "combined" / (key + ".jpg")))
            # plt.show()
        
    def get_distribution(self):
        '''
        Get distribution of classes in dataset

        Returns:
        distribution (dict): distribution of classes in dataset
        '''
        distribution = {}
        for key in self.classes:
            distribution[key] = len(self.df[self.df["dx"] == key])
        
        return distribution

    def balance(self):
        '''
        Balance dataset by oversampling minority classes

        Returns:
        df (DataFrame): balanced dataset
        '''
        max_class = max(self.get_distribution(), key=self.get_distribution().get)
        max_class_count = self.get_distribution()[max_class]

        df = self.df.copy()

        for key in self.classes:
            if key != max_class:
                count = max_class_count - self.get_distribution()[key]
                class_df = self.df[self.df["dx"] == key]
                sampled_df = class_df.sample(count, replace=True, random_state=self.seed)
                self.df = pd.concat([self.df, sampled_df])
        
        return df

    def __getitem__(self, index):
        '''
        Get item from dataset

        Parameters:
        index (int): index of item to get
        '''

        if self.testing:
            return self.get_test_item(index)
        else:
            return self.get_train_item(index)

        
    def get_test_item(self, index):
        '''
        Get item from dataset without augmentation

        Parameters:
        index (int): index of item to get
        '''
        result = self.df.iloc[index]

        filename = result["image_id"] + ".jpg"

        path = self.path / "combined" / filename

        image = Image.open(path)

        # resize image to 128x128
        # NOTE: Image cropping should be done to square of maximum size that fits image in center of image for testing data
        
        image = self.static_crop(image)

        size = 256
        image = image.resize((size, size))

        gt = torch.zeros(len(self.classes))
        gt[self.classes[result["dx"]]] = 1.0

        return torch.from_numpy(np.array(image).transpose((2, 0, 1)))/255, gt, result
    
    def get_train_item(self, index):
                
        result = self.df.iloc[index]

        filename = result["image_id"] + ".jpg"

        path = self.path / "combined" / filename

        image = Image.open(path)

        if self.do_augment:
            image = self.augment(image)

        # resize image to 128x128
        # NOTE: Image cropping should be done to square of maximum size that fits image in center of image for testing data
        size = 256
        image = image.resize((256, 256))

        gt = torch.zeros(len(self.classes))
        gt[self.classes[result["dx"]]] = 1.0

        if self.testing:
            return torch.from_numpy(np.array(image).transpose((2, 0, 1)))/255, gt, self.pigment_dict[result["image_id"]]
        else:
            return torch.from_numpy(np.array(image).transpose((2, 0, 1)))/255, gt
        
    def static_crop(self, image):
        '''
        crop to center of image
        '''
        width, height = image.size
        
        new_size = min(width, height)
        left = width // 2 - new_size // 2
        top = height // 2 - new_size // 2
        right = left + new_size
        bottom = top + new_size
        
        return image.crop((left, top, right, bottom))
    
    def crop(self, image):
        '''
        Crop image to a square that takes up at least 70% of the original image

        Parameters:
        image (PIL Image): image to crop

        Returns:
        image (PIL Image): cropped image
        '''
        width, height = image.size

        crop_percent = np.random.rand() * 0.3 + 0.7

        new_size = int(min(width, height) * crop_percent)
        left = np.random.randint(0, width - new_size + 1)
        top = np.random.randint(0, height - new_size + 1)
        right = left + new_size
        bottom = top + new_size

        return image.crop((left, top, right, bottom))
    
    def augment(self, image):
        '''
        Augment image with random transformations

        Parameters:
        image (PIL Image): image to augment

        Returns:
        image (PIL Image): augmented image
        '''
        # plt.imshow(image)
        # plt.show()

        # crop
        image = self.crop(image)

        # horizontal flip
        if np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # vertical flip
        if np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # rotation 180 degrees
        if np.random.rand() > 0.5:
            image = image.rotate(180)

        #brightness
        brightness = 0.5
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1 + (np.random.rand() * brightness - brightness/2))
        
        #contrast
        contrast = 0.5
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1 + (np.random.rand() * contrast - contrast/2))

        #saturation
        saturation = 0.5
        if np.random.rand() > 0.5:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1 + (np.random.rand() * saturation - saturation/2))
        
        #blur
        blur = 0.5
        if np.random.rand() > 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.rand() * blur))

        # plt.imshow(image)
        # plt.show()

        return image

class train_test():
    def __init__(self, path, seed=42):
        self.path = Path(path)
        self.dataframe = pd.read_csv(self.path / "HAM10000_metadata.csv")
        self.seed = seed

    def getSplit(self, test_size=0.2, seed=42):
        '''
        Split dataset into training and testing sets

        Parameters:
        test_size (float): size of test set out of 1 (default 0.2)

        Returns:
        train (DataFrame): training set
        test (DataFrame): testing set
        '''
        train_df, test_df = train_test_split(self.dataframe, test_size=test_size, random_state=seed)
        train = HAM10000(train_df, self.path,augment=True, equalize=True, testing=False)
        test = HAM10000(test_df, self.path, augment=False, equalize=False, testing=True)

        return {"train": train, "test": test}

if __name__ == "__main__":
    path = "HAM10000/archive/"
    dataset = train_test(path)
    datasets_dict = dataset.getSplit()
    train = datasets_dict["train"]
    test = datasets_dict["test"]

    test.check_threshold()

    image, gt = train[0]