import argparse
import csv
from pathlib import Path
import random
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10, MNIST, SVHN, CelebA, FashionMNIST
import torchvision
import torch


def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]

class CIFAR10C(torchvision.datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None, corruptions_file=None, severity=1):
        corruptions = load_txt(corruptions_file)

        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)[(severity-1)*10000:severity*10000]
        self.targets = np.load(target_path)[(severity-1)*10000:severity*10000]

        
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


def extract_subset(dataset, num_subset :int, random_subset :bool):
    if random_subset:
        random.seed(0)
        indices = random.sample(list(range(len(dataset))), num_subset)
    else:
        indices = [i for i in range(num_subset)]
        
    return torch.utils.data.Subset(dataset, indices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )
    parser.add_argument(
        "--download_celeba",
        type=bool,
        default=True,
        help="Will attempt to download the CelebA dataset." " Set to False if manually downloaded.",
    )
    args = parser.parse_args()
    return args


def download_data(data_root, download_celeba):
    # MNIST
    MNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = MNIST(root=data_root, train=True if set == "train" else False)
        dataset_name = dataset.__class__.__name__
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # FashionMNIST
    FashionMNIST(data_root, download=True)
    for set in ["train", "test"]:
        dataset = FashionMNIST(root=data_root, train=True if set == "train" else False)
        dataset_name = dataset.__class__.__name__
        out_dir = Path(dataset.raw_folder).parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img)
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # CIFAR10
    root = Path(data_root) / "CIFAR10" / "raw"
    CIFAR10(root, download=True)
    for set in ["train", "test"]:
        dataset = CIFAR10(root=root, train=True if set == "train" else False)
        dataset_name = dataset.__class__.__name__
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # CIFAR10C
    root = Path(data_root) / "CIFAR10C" / "raw"
    curruption_list = load_txt(r"/media/chris/My Passport/Philips/Anomaly/CIFAR/corruptions")
    # CIFAR10(root, download=True)
    for set in ["test"]:
        for corruption in curruption_list:
            if corruption == "natural":
                continue
            for severity in range(1,6): 
                dataset = CIFAR10C(r"/media/chris/My Passport/Philips/Anomaly/CIFAR/CIFAR-10-C", corruption, transform=None, corruptions_file=r"/media/chris/My Passport/Philips/Anomaly/CIFAR/corruptions", severity=severity)
        
                # dataset = CIFAR10C(root=root, train=True if set == "train" else False)
                dataset_name = dataset.__class__.__name__ + "_" + corruption+"_"+str(severity)
                out_dir = root.parent / "numpy" / set / corruption / str(severity)
                out_dir.mkdir(parents=True, exist_ok=True)
                for i in range(len(dataset)):
                    img, label = dataset[i]
                    img_np = np.array(img).transpose((2, 0, 1))
                    np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # SHVN
    root = Path(data_root) / "SVHN" / "raw"
    for set in ["train", "test"]:
        dataset = SVHN(root=root, split=set, download=True)
        dataset_name = dataset.__class__.__name__
        out_dir = root.parent / "numpy" / set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img, label = dataset[i]
            img_np = np.array(img).transpose((2, 0, 1))
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # # CelebA
    # root = Path(data_root) / "CelebA" / "raw"
    # for set in ["train", "valid", "test"]:
    #     dataset = CelebA(root=root, split=set, download=download_celeba)
    #     dataset_name = dataset.__class__.__name__
    #     out_dir = root.parent / "numpy" / set
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     for i in range(len(dataset)):
    #         img, label = dataset[i]
    #         img = img.resize((32, 32))
    #         img_np = np.array(img).transpose((2, 0, 1))
    #         np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)


def create_train_test_splits(data_root):
    splits_dir = Path(data_root) / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    # need to create a train/val split for these datasets
    # for dataset in ["FashionMNIST", "MNIST", "CIFAR10", "SVHN"]:
    #     numpy_data_root = Path(data_root) / dataset / "numpy"
    #     train_and_val_list = list((numpy_data_root / "train").glob("*"))
    #     train_list, val_list = train_test_split(train_and_val_list, test_size=0.05, random_state=42)
    #     test_list = list((numpy_data_root / "test").glob("*"))
    #     for split_name, data_split in zip(
    #         ["train", "val", "test"], [train_list, val_list, test_list]
    #     ):
    #         save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")

    root = Path(data_root) / "CIFAR10C" / "raw"
    curruption_list = load_txt(r"/media/chris/My Passport/Philips/Anomaly/CIFAR/corruptions")
    # CIFAR10(root, download=True)
    dataset = 'CIFAR10C'
    for set in ["test"]:
        for corruption in curruption_list:
            if corruption == "natural":
                continue
            for severity in range(1,6): 
                numpy_data_root = Path(data_root) / dataset / "numpy" / "test"/ corruption / str(severity)

                test_list = list((numpy_data_root).glob("*"))
                for split_name, data_split in zip(
                    ["test"], [test_list]
                ):
                    save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}_{corruption}_{severity}.csv")

    # CelebA already has a train/val split
    dataset = "CelebA"
    numpy_data_root = Path(data_root) / dataset / "numpy"
    train_list = list((numpy_data_root / "train").glob("*"))
    val_list = list((numpy_data_root / "valid").glob("*"))
    test_list = list((numpy_data_root / "test").glob("*"))
    for split_name, data_split in zip(["train", "val", "test"], [train_list, val_list, test_list]):
        save_list_as_csv(data_split, splits_dir / f"{dataset}_{split_name}.csv")


if __name__ == "__main__":
    args = parse_args()
    # download_data(data_root=args.data_root, download_celeba=args.download_celeba)
    create_train_test_splits(data_root=args.data_root)
