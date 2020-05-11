import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class Dataset(Dataset):
    def __init__(self, dataset_path, class_path, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_class_labels(class_path)
        self.sequences = self._extract_sequence_paths(dataset_path)
        self.sequence_length = sequence_length
        self.label_names = list(self.label_index.keys())
        self.num_classes = len(self.label_names)
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _extract_class_labels(self, class_path):
        """ Extract class labels"""
        classes = {}
        for line in open(class_path, "r"):
            idx, cl = line.rstrip('\n').split(' ',1)
            classes[int(idx)] = cl
        return classes
        

    def _extract_sequence_paths(self, dataset_path):
        """ Extracts paths to sequences given the specified train / test split """
        return glob.glob(os.path.join(dataset_path,"*_rgb.npy"))

    def _extract_name(self, sequence):
        return sequence.split('/')[-1].rstrip('_rgb.npy')

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return int(path.split('_',1)[-1].lstrip('a').split('_')[0])-1

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        n_frame = len(sequence)
        right_pad = sequence[n_frame-1]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence[n_frame]=right_pad
                n_frame+=1
        return sequence

    def _transform(self, a, k):
        size = a.shape
        out = np.zeros((k,size[1],size[2]))
        for i in range (k):
            out[i,:,:]=a[int(i/(k/len(a))),:,:]
        out = np.array(out, dtype=np.float32)
        return(out)

        
    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # load video frame data
        video_data = np.load(sequence_path)
        video_name = self._extract_name(sequence_path)
        video_data = video_data.item()
        # Pad frames sequences shorter than `self.sequence_length` to length
        video_data = self._pad_to_length(video_data)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = len(video_data) // self.sequence_length
            start_i = np.random.randint(0, len(video_data) - sample_interval * self.sequence_length + 1)
            flip = np.random.random() < 0.5
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else len(video_data) // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, len(video_data), sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.fromarray(np.uint8(video_data[i])))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_tensor=np.array(image_tensor, dtype=np.float32)
                image_tensor=torch.tensor(image_tensor) 
                image_sequence.append(image_tensor)
  
        image_sequence = torch.stack(image_sequence)
        target = self._activity_from_path(video_name)

        #pose_data = np.load(sequence_path.rstrip('_rgb.npy')+'_skeleton.npy')
        #if start_i == 0:
        #    pose = pose_data[start_i:(start_i+sample_interval*self.sequence_length):sample_interval]
        #else:
        #    pose = pose_data[(start_i-1):(start_i+sample_interval*self.sequence_length-1):sample_interval]
        #pose = torch.tensor(pose, dtype=torch.float32)
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
