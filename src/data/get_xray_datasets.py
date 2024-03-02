import argparse
import csv
from pathlib import Path
import numpy as np
import pydicom
import os
import json
from dataclasses import dataclass
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root", type=str, default="None", help="Directory data will be stored."
    )

    args = parser.parse_args()
    return args


def download_data(data_root):
    splits_dir = Path(data_root) / "data_splits"
    splits_dir.mkdir(exist_ok=True, parents=True)

    base_dir = r"/media/chris/My Passport/Philips/data_2023/Anomaly"
    json_path = os.path.join(base_dir, "data_description_clock.json")

    dataset_0 = ComposedXrayImageDataset(base_dir, json_path, None, series_per_mode=1, overlap=0.1, patch_size=128, images_per_series=200, transform=None, modes_to_exclude=list(np.arange(0, 17)))

    dataset_size = len(dataset_0)
    split = int(np.floor(0.2 * dataset_size))
    train_size = int(dataset_size - split)
    test_size = int(np.floor(split/2))
    val_size = int(dataset_size - test_size-train_size)

    train_set, val_set0, test_set0 = torch.utils.data.random_split(dataset_0, [train_size, val_size, test_size])

    for set, dataset in zip(["training", "validation", "test"], [train_set, val_set0, test_set0]):
        dataset_name = set
        out_dir = data_root / "xray" / "numpy"/ set
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(len(dataset)):
            img = dataset[i]
            img_np = img.img
            np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

        save_list_as_csv(
            [str(item.img) for item in dataset],
            splits_dir / f"XRAY_{set}.csv",
        )

    # # CIFAR10(root, download=True)
    # for set in ["test"]:
    #     for corruption in curruption_list:
    #         if corruption == "natural":
    #             continue
    #         for severity in range(1,6): 
    #             dataset = CIFAR10C(r"/media/chris/My Passport/Philips/Anomaly/CIFAR/CIFAR-10-C", corruption, transform=None, corruptions_file=r"/media/chris/My Passport/Philips/Anomaly/CIFAR/corruptions", severity=severity)
        
    #             # dataset = CIFAR10C(root=root, train=True if set == "train" else False)
    #             dataset_name = dataset.__class__.__name__ + "_" + corruption+"_"+str(severity)
    #             out_dir = root.parent / "numpy" / set / corruption / str(severity)
    #             out_dir.mkdir(parents=True, exist_ok=True)
    #             for i in range(len(dataset)):
    #                 img, label = dataset[i]
    #                 img_np = np.array(img).transpose((2, 0, 1))
    #                 np.save(out_dir / f"{dataset_name}_{i}.npy", img_np)

    # for split in ["training", "validation", "test"]:
    #     dataset = MedNISTDataset(
    #         root_dir=args.data_root, section=split, download=True, progress=False, seed=0
    #     )
    #     data_list = dataset.data
    #     datasets = set(item["class_name"] for item in data_list)
    #     for dataset in datasets:
    #         dataset_list = [item["image"] for item in data_list if item["class_name"] == dataset]
    #         save_list_as_csv(
    #             dataset_list,
    #             splits_dir / f"{dataset}_{split.replace('ing','').replace('idation','')}.csv",
    #         )
        print("debug")


def save_list_as_csv(list, output_path):
    with open(output_path, "w", newline="") as f:
        tsv_output = csv.writer(f, delimiter=",")
        tsv_output.writerow(list)



@dataclass
class FilteredImageData:
    low_img: np.ndarray
    high_img: np.ndarray
    fluoro_flavor: np.int8
    exposure: np.bool8
    sid: np.int16

    def __init__(self, low_img, high_img, fluoro_flavor, exposure, sid):
        self.low_img = low_img
        self.high_img = high_img
        self.fluoro_flavor = fluoro_flavor
        self.exposure = exposure
        self.sid = sid

@dataclass
class ImageData:
    img: np.ndarray
    fluoro_flavor: np.int8
    exposure: np.bool8
    sid: np.int16

    def __init__(self, img, fluoro_flavor, exposure, sid):

        self.img = img
        self.fluoro_flavor = fluoro_flavor
        self.exposure = exposure
        self.sid = sid

def get_code(flavor, exposure, sid):
    
    # •	Flavor: first, SID 90
    # •	Flavor: second, SID 90
    # •	Flavor: third, SID 90
    # •	Flavor: first, SID 100
    # •	Flavor: second, SID 100
    # •	Flavor: third, SID 100
    # •	Flavor: first, SID 110
    # •	Flavor: second, SID 110
    # •	Flavor: third, SID 110

    # •	Exposure: first, SID 90
    # •	Exposure: second, SID 90
    # •	Exposure: third, SID 90
    # •	Exposure: first, SID 100
    # •	Exposure: second, SID 100
    # •	Exposure: third, SID 100
    # •	Exposure: first, SID 110
    # •	Exposure: second, SID 110
    # •	Exposure: third, SID 110
    
    sid = int(sid/10)

    if flavor == 0 and exposure == 0 and sid == 90:
        return 0
    elif flavor == 1 and exposure == 0 and sid == 90:
        return 1
    elif flavor == 2 and exposure == 0 and sid == 90:
        return 2
    elif flavor == 0 and exposure == 0 and sid == 100:
        return 3
    elif flavor == 1 and exposure == 0 and sid == 100:
        return 4
    elif flavor == 2 and exposure == 0 and sid == 100:
        return 5
    elif flavor == 0 and exposure == 0 and sid == 110:
        return 6
    elif flavor == 1 and exposure == 0 and sid == 110:
        return 7
    elif flavor == 2 and exposure == 0 and sid == 110:
        return 8
    elif flavor == 0 and exposure == 1 and sid == 90:
        return 9
    elif flavor == 1 and exposure == 1 and sid == 90:
        return 10
    elif flavor == 2 and exposure == 1 and sid == 90:
        return 11
    elif flavor == 0 and exposure == 1 and sid == 100:
        return 12
    elif flavor == 1 and exposure == 1 and sid == 100:
        return 13
    elif flavor == 2 and exposure == 1 and sid == 100:
        return 14
    elif flavor == 0 and exposure == 1 and sid == 110:
        return 15
    elif flavor == 1 and exposure == 1 and sid == 110:
        return 16
    elif flavor == 2 and exposure == 1 and sid == 110:
        return 17
    else:
        return -1





class ComposedXrayImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, json_description, image_filter=None, transform = None, patch_size: int = 128, overlap: float = 0.25, series_per_mode=6, images_per_series = None, preload=True, modes_to_exclude = []):

        self.base_dir = base_dir
        self.transform = transform
        self.overlap = overlap
        self.patch_size = patch_size
        self.data = list()
        self.filter = image_filter
        self.series_per_mode = series_per_mode

        
        data_description = open(json_description)
        data_description = json.load(data_description)

        total_count = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]

        print(base_dir)

        if preload:

            for item in data_description.keys():
                series_data = data_description[item]

                flavor, exposure, sid = series_data['flavor'], series_data['fluo/exposure'], series_data['sid']
                s = get_code(flavor, exposure, sid)
                if s in modes_to_exclude:
                    continue
                if total_count[s] < series_per_mode:
                    total_count[s] += 1
                    print(f'Adding series {s}')
                    path = base_dir + series_data['path']
                    series = pydicom.dcmread(path)
                    series_images = series.pixel_array
                    _, img_h, img_w  = series_images.shape

                    X_points = self.start_points(img_w, self.patch_size, self.overlap)
                    Y_points = self.start_points(img_h, self.patch_size, self.overlap)
                    
                    for idx, image in enumerate(series_images):
                        if images_per_series is not None and idx >= images_per_series:
                            break
                        image = image/np.power(2, 12) # Normalize images between 0 and 1
                        if self.filter is not None:
                            low, high = self.filter(image)

                            low_patches, count = self.split_image_with_points(low, X_points, Y_points)
                            high_patches, count = self.split_image_with_points(high, X_points, Y_points)

                            for patch_idx in range(count):
                                self.data.append(FilteredImageData(low_img = low_patches[patch_idx], high_img = high_patches[patch_idx], fluoro_flavor = flavor , exposure = exposure, sid = sid))

                        else:
                            image_patches, count = self.split_image_with_points(image, X_points, Y_points)

                            for patch_idx in range(count):
                                self.data.append(ImageData(img = image_patches[patch_idx], fluoro_flavor = flavor , exposure = exposure, sid = sid))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]) if self.transform else self.data[idx]


    def split_image(self, image):
        img_h, img_w, img_c,  = image.shape
        X_points = self.start_points(img_w, self.patch_size, self.overlap)
        Y_points = self.start_points(img_h, self.patch_size, self.overlap)
        count = 0
        frmt = "png"
        patches = np.empty((len(Y_points)*len(X_points), self.patch_size, self.patch_size, img_c))
        for i in Y_points:
            for j in X_points:
                split = image[i:i+self.patch_size, j:j+self.patch_size, :]
                patches[count,:, :, :] = split
                #cv2.imwrite('patch_{}.{}'.format( count, frmt), split)
                count += 1
        return patches

    def split_image_with_points(self, image, X_points, Y_points):

        count = 0

        patches = np.empty((len(Y_points)*len(X_points), self.patch_size, self.patch_size))
        for i in Y_points:
            for j in X_points:
                split = image[i:i+self.patch_size, j:j+self.patch_size]
                patches[count,:, :] = split
                #cv2.imwrite('patch_{}.{}'.format( count, frmt), split)
                count += 1
        return patches, count

    
    def start_points(self, size, patch_size, overlap=0):
        points = [0]
        stride = int(patch_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + patch_size >= size:
                points.append(size - patch_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points
    




if __name__ == "__main__":
    args = parse_args()
    download_data(data_root=args.data_root)
