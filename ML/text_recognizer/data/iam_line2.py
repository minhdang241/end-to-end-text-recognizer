from pathlib import Path
from typing import Union, List
import argparse
import json
import random
import os

from PIL import Image, ImageFile, ImageOps
import numpy as np
import torch
from torchvision import transforms

from text_recognizer.data.util import BaseDataset, convert_strings_to_labels
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam import IAM

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_lines2"
META_FILE_PATH = BaseDataModule.data_dirname()/"downloaded/iam/iamdb/ascii/sentences.txt"
TRAIN_FRAC = 0.8
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 800 # Rounding up the actual empirical max to a power of 2
MAX_LENGTH = 42
IAM_ESSENTIALS = Path(__file__).parents[0].resolve()/"iam_essentials.json"
DATA_JSON = Path(__file__).parents[0].resolve()/"data.json"
IMAGES_PATH = BaseDataModule.data_dirname() /"sentences"

class IAMLines2(BaseDataModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "false") == "true"
        self.max_length = self.args.get("max_length", MAX_LENGTH)
        self.mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " ", "!", "\"", "#", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "?", "<P>","<B>"]
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)  # We assert that this is correct in setup()
        self.output_dims = (self.max_length, 1)  # We assert that this is correct in setup()
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        # Check if the data json file exist
        if not os.path.isfile(DATA_JSON):
            paths_labels = create_data_file(META_FILE_PATH, max_length=self.max_length)
    
    def setup(self, stage:str = None):            
        with open(DATA_JSON) as json_file:
            paths_labels = json.load(json_file)
        
        list_paths_labels = [(key, value) for key, value in paths_labels.items()]

        random.shuffle(list_paths_labels) 
        shuffled_data = list_paths_labels
        trainval_data = shuffled_data[:int(len(list_paths_labels)*0.9)] 
        test_data = shuffled_data[int(len(list_paths_labels)*0.9):]  
        
        if stage == "fit" or stage is None:
            filenames_trainval = [IMAGES_PATH/trainval[0] for trainval in trainval_data]
            labels_trainval = [trainval[1] for trainval in trainval_data]
            x_trainval = [Image.open(filename) for filename in filenames_trainval]
            y_trainval = convert_strings_to_labels(labels_trainval, self.inverse_mapping, length=self.output_dims[0])
            data_trainval = BaseDataset(x_trainval, y_trainval, transform=get_transform(IMAGE_WIDTH, self.augment))
            train_size = int(TRAIN_FRAC * len(data_trainval))
            val_size = len(data_trainval) - train_size
            self.data_train, self.data_val = torch.utils.data.random_split(
                data_trainval, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        if stage == "test" or stage is None:
            filenames_test = [IMAGES_PATH/test[0] for test in test_data]
            labels_test = [test[1] for test in test_data]
            x_test = [Image.open(filename) for filename in filenames_test]
            y_test = convert_strings_to_labels(labels_test, self.inverse_mapping, length=self.output_dims[0])
            self.data_test = BaseDataset(x_test, y_test, transform=get_transform(IMAGE_WIDTH))

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        parser.add_argument("--max_length", type=int, defualt=MAX_LENGTH)
        return parser
    
    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Lines Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def get_transform(image_width, augment=False):
    """Augment with brightness, slight rotation, slant, translation, scale, and Gaussian noise."""
    def embed_crop(crop, augment=augment, image_width=image_width):
        image = Image.new("L", (image_width, IMAGE_HEIGHT))

        # Resize crop
        crop_width, crop_height = crop.size
        new_crop_height = IMAGE_HEIGHT
        new_crop_width = int(new_crop_height / crop_height * crop_width)
        if augment:
            # Add random stretching
            new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
            new_crop_width = min(new_crop_width, image_width)
        crop_resized = crop.resize((new_crop_width, new_crop_height), resample=Image.BILINEAR)

        # Embed in the image
        x, y = 28, 0
        image.paste(crop_resized, (x, y))

        return image

    transforms_list = [transforms.Lambda(embed_crop)]
    if augment:
        transforms_list += [
            transforms.ColorJitter(brightness=(0.8, 1.6)),
            transforms.RandomAffine(
                degrees=1,
                shear=(-30, 20),
                resample=Image.BILINEAR,
            ),
        ]
    transforms_list += [
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x - 0.5)
    ]
    return transforms.Compose(transforms_list)


def create_data_file(file_path: Union[str, Path], max_length=None) -> None:
    with open(file_path) as f:
        lines = f.readlines()
    
    paths_labels = dict()
    
    for line in lines:
        # ignore the comments
        if not line or line[0] == "#":
            continue

        tokens = line.strip().split(' ')
        assert len(tokens) >= 9
        # Get labels and filepaths
        label = truncate_label(" ".join(' '.join(tokens[9:]).split("|")), max_length=max_length)
        
        path_tokens = tokens[0].split("-")

        image_path = f"{path_tokens[0]}/{path_tokens[0]}-{path_tokens[1]}/{tokens[0]}.png"
        paths_labels[image_path] = label
    with open(DATA_JSON, 'w') as json_file:
        json.dump(paths_labels, json_file)
    return paths_labels


def truncate_label(text:str, max_length:int=None):
    # ctc_loss can't compute loss if it cannot find a mapping between text label and input
    # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
    # If a too-long label is provided, ctc_loss returns an infinite gradient
    if max_length is None:
        return text

    cost = 0
    for i in range(len(text)):
        if i != 0 and text[i] == text[i - 1]:
            cost += 2
        else:
            cost += 1
        if cost > max_length:
            return text[:i]
    return text