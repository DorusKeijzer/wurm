import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class COCODataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            annotation_file (str): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied
                                            on a sample (image and target).
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load the COCO annotation file (JSON)
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # Extract useful information
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']

        # Group annotations by image_id for easy access
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image info and corresponding annotations
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.img_to_anns.get(img_info['id'], [])

        # Extract bounding boxes and categories
        bboxes = []
        categories = []
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]
            bboxes.append(bbox)
            categories.append(ann['category_id'])

        target = {
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'categories': torch.tensor(categories, dtype=torch.int64)
        }

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, target


transform = transforms.Compose([
    transforms.ToTensor()
])


proof_of_concept_path_train = "./proof_of_concept/train/"

train_annotation_file = "./proof_of_concept/train/_annotations.coco.json"

train_dataset = COCODataset(img_dir=proof_of_concept_path_train,
                            annotation_file=train_annotation_file, transform=transform)

train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

proof_of_concept_path_test = "./proof_of_concept/test/"

test_annotation_file = "./proof_of_concept/test/_annotations.coco.json"

test_dataset = COCODataset(img_dir=proof_of_concept_path_test,
                           annotation_file=test_annotation_file, transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True,
                             collate_fn=lambda x: tuple(zip(*x)))

proof_of_concept_path_val = "./proof_of_concept/val/"

val_annotation_file = "./proof_of_concept/val/_annotations.coco.json"

val_dataset = COCODataset(img_dir=proof_of_concept_path_val,
                          annotation_file=val_annotation_file, transform=transform)

val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True,
                            collate_fn=lambda x: tuple(zip(*x)))
