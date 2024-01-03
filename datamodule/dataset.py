import os
import cv2
import glob
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path: str, transforms=None, is_train: bool = True):
        self.transforms = transforms
        self.is_train = is_train
        with open(data_path + "/wnids.txt", "r") as f:
            self.label_list = f.read().splitlines()

        if is_train:
            self.images = glob.glob(data_path + "/train/*/images/*.JPEG")
            self.train_list = dict()
            for image in self.images:
                label = image.split(os.sep)[-3]
                self.train_list[image] = self.label_list.index(label)

        else:
            self.images = glob.glob(data_path + "/val/images/*.JPEG")
            self.val_list = dict()
            with open(data_path + "/val/val_annotations.txt", "r") as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split("\t")
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        print(image_file)
        image = cv2.imread(image_file)
        if self.is_train:
            label = self.train_list[image_file]
        else:
            label = self.val_list[os.path.basename(image_file)]
        transformed_image = self.transforms(image=image)["image"]
        return transformed_image, label


if __name__ == "__main__":
    import albumentations as A

    data_dir = "/workspace/torch_backbone/data/tiny-imagenet-200"
    transforms = A.Compose(
        [
            A.Resize(256, 256),
        ]
    )
    dataset = CustomDataset(data_dir, transforms=transforms)
    sample = dataset.__getitem__(0)
    sample_image = sample[0]
    os.makedirs("/workspace/torch_backbone/figures", exist_ok=True)
    cv2.imwrite("/workspace/torch_backbone/figures/sample.jpg", sample_image)
