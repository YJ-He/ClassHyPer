import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

rgb_mean = (0.485, 0.456, 0.406)
rgb_std = (0.229, 0.224, 0.225)

class MyDataset(Dataset):
    def __init__(self,
                 config,
                 args,
                 subset,
                 file_length=None):
        super(MyDataset, self).__init__()
        assert subset == 'train' or subset == 'val' or subset == 'test' or subset == 'train_unsup'

        self.args = args
        self.config = config
        self.root = args.input
        self.subset = subset
        self.data = self.config.data_folder_name  # image
        self.target = self.config.target_folder_name  # label
        self._file_length = file_length

        if self.config.nb_classes == 2:  # binary class(buildings and roads)
            self.mapping = {
                0: 0,
                255: 1,
            }
            self.class_names = ['other', 'building']
        elif self.config.nb_classes == 6:  # ISPRS dataset (six class)
            self.mapping = {
                (255, 255, 255): 0,
                (0, 0, 255): 1,
                (0, 255, 255): 2,
                (0, 255, 0): 3,
                (255, 255, 0): 4,
                (255, 0, 0): 5,
                (0, 0, 0): 6
            }
            self.class_names = ['Impervious surfaces', 'building', 'low vegetation', 'tree', 'car',
                                'clutter/background']

        self.data_list = []
        self.target_list = []
        with open(os.path.join(self.root, subset + '_image.txt'), 'r') as f:
            for line in f:
                if line.strip('\n') != '':
                    self.data_list.append(line.strip('\n'))
        if subset != 'train_unsup':
            with open(os.path.join(self.root, subset + '_label.txt'), 'r') as f:
                for line in f:
                    if line.strip('\n') != '':
                        self.target_list.append(line.strip('\n'))

            if self._file_length is not None:
                self.data_list, self.target_list = self._construct_new_file_list(self._file_length, is_UnsupData=False)
        else:
            if self._file_length is not None:
                self.data_list = self._construct_new_file_list(self._file_length, is_UnsupData=True)

    def _construct_new_file_list(self, length, is_UnsupData):
        """
        Construct new file list based on whether is unlabeled data or not
        """
        assert isinstance(length, int)
        files_len = len(self.data_list)

        if length < files_len:
            if not is_UnsupData:
                return self.data_list[:length], self.target_list[:length]
            else:
                return self.data_list[:length]

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_data_list = self.data_list * (length // files_len)
        new_data_list += [self.data_list[i] for i in new_indices]

        if not is_UnsupData:
            new_target_list = self.target_list * (length // files_len)
            new_target_list += [self.target_list[i] for i in new_indices]
            return new_data_list, new_target_list
        else:
            return new_data_list

    def mask_to_class(self, mask):
        """
        Encode class to number
        """
        if self.config.nb_classes == 2:
            for k in self.mapping:
                mask[mask == k] = self.mapping[k]
            return mask
        elif self.config.nb_classes == 6:
            mask = mask.permute(2, 0, 1).contiguous()
            m = torch.empty(self.config.input_size, self.config.input_size, dtype=torch.long)
            for k in self.mapping:
                idx = (mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
                validx = (idx.sum(0) == 3)
                m[validx] = torch.tensor(self.mapping[k], dtype=torch.long)
            return m

    def train_transforms(self, image, mask):
        """
        Preprocessing and augmentation on training data (image and label)
        """
        in_size = self.config.input_size
        train_transform = A.Compose(
            [
                A.Resize(in_size, in_size, interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8),
                A.RandomRotate90(p=0.8),
                A.Transpose(p=0.8),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = train_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        mask = self.mask_to_class(mask)

        mask = mask.float()
        return image, mask

    def untrain_transforms(self, image, mask):
        """
        Preprocessing on val or test data (image and label)
        """
        untrain_transform = A.Compose(
            [
                A.Resize(self.config.eval_size, self.config.eval_size, interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = untrain_transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        mask = self.mask_to_class(mask)
        mask = mask.float()
        return image, mask

    def untrain_transforms1(self, image):
        """
        Preprocessing on unsup data (image)
        """
        untrain_transform = A.Compose(
            [
                A.Resize(self.config.eval_size, self.config.eval_size),
                A.Normalize(mean=rgb_mean, std=rgb_std),
                ToTensorV2(),
            ]
        )
        transformed = untrain_transform(image=image)
        image = transformed["image"]

        return image

    def __getitem__(self, index):
        image = cv2.imread(self.data_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.subset != 'train_unsup':
            mask = np.array(Image.open(self.target_list[index])).astype(np.uint8)

        if self.subset == 'train':
            if not self.args.is_test:
                t_datas, t_targets = self.train_transforms(image, mask)
            else:
                t_datas, t_targets = self.untrain_transforms(image, mask)
            return t_datas, t_targets, self.data_list[index]
        elif self.subset == 'train_unsup':
            t_datas = self.untrain_transforms1(image)
            return t_datas, self.data_list[index]
        elif self.subset == 'val' or self.subset == 'test':
            t_datas, t_targets = self.untrain_transforms(image, mask)
            return t_datas, t_targets, self.data_list[index]

    def __len__(self):

        return len(self.data_list)
