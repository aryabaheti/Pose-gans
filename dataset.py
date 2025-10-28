import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class SamePersonPairsDataset(Dataset):
    def __init__(self, pairs_csv, img_dir, pose_dir=None, transform=None):
        """
        pairs_csv: Path to CSV file with pairs of filenames (src, tgt)
        img_dir: Directory with the images
        pose_dir: Directory with pose heatmaps (optional)
        transform: torchvision transforms for images
        """
        self.img_dir = img_dir
        self.pose_dir = pose_dir

        # Read pairs from CSV file
        self.pairs = []
        with open(pairs_csv, 'r') as f:
            for line in f:
                src, tgt = line.strip().split(',')
                self.pairs.append((src, tgt))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_name, tgt_name = self.pairs[idx]

        # Load source and target images
        src_path = os.path.join(self.img_dir, src_name)
        tgt_path = os.path.join(self.img_dir, tgt_name)

        src_img = Image.open(src_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')

        src_img = self.transform(src_img)
        tgt_img = self.transform(tgt_img)

        data = {
            'src_img': src_img,
            'tgt_img': tgt_img,
            'src_name': src_name,
            'tgt_name': tgt_name,
        }

        # If pose heatmaps are available, load them too
        if self.pose_dir is not None:
            src_pose_path = os.path.join(self.pose_dir, os.path.splitext(src_name)[0] + '.png')  # or .npy
            tgt_pose_path = os.path.join(self.pose_dir, os.path.splitext(tgt_name)[0] + '.png')
            
            if os.path.exists(src_pose_path):
                src_pose = Image.open(src_pose_path).convert('L')
                src_pose = transforms.ToTensor()(src_pose)
                data['src_pose'] = src_pose
            if os.path.exists(tgt_pose_path):
                tgt_pose = Image.open(tgt_pose_path).convert('L')
                tgt_pose = transforms.ToTensor()(tgt_pose)
                data['tgt_pose'] = tgt_pose

        return data
