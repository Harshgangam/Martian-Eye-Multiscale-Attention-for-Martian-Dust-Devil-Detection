import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np

class MarsDustDevilDataset(Dataset):
    """
    Dataset loader for Martian Dust Devil Detection in COCO format.
    Handles train, validation, and test splits with annotations.
    """
    def __init__(self, data_dir, split='train', img_size=(416, 416), transform=None):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.split_dir = os.path.join(data_dir, split)
        self.ann_file = os.path.join(self.split_dir, '_annotations.coco.json')
        
        self.images = []
        self.annotations_map = {}
        
        if os.path.exists(self.ann_file):
            with open(self.ann_file, 'r') as f:
                coco_data = json.load(f)
                
            img_dict = {img['id']: img for img in coco_data.get('images', [])}
            
            for ann in coco_data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in self.annotations_map:
                    self.annotations_map[img_id] = []
                self.annotations_map[img_id].append(ann['bbox']) # [x, y, w, h]
                
            for img_id, img_info in img_dict.items():
                file_name = img_info['file_name']
                img_path = os.path.join(self.split_dir, file_name)
                if os.path.exists(img_path):
                    has_obj = img_id in self.annotations_map
                    bboxes = self.annotations_map.get(img_id, [])
                    self.images.append({
                        'img_id': img_id,
                        'file_name': file_name,
                        'path': img_path,
                        'width': img_info.get('width', 512),
                        'height': img_info.get('height', 512),
                        'bboxes': bboxes,
                        'has_dust_devil': int(len(bboxes) > 0)
                    })
        
        self.transform = transform or T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        img = Image.open(item['path']).convert('RGB')
        orig_w, orig_h = item['width'], item['height']
        
        # Transform image
        img_tensor = self.transform(img)
        
        # Process bounding boxes
        normalized_boxes = []
        for bbox in item['bboxes']:
            x, y, w, h = bbox
            # Normalize to [0, 1] relative to original image size
            norm_box = [x / orig_w, y / orig_h, (x + w) / orig_w, (y + h) / orig_h]
            normalized_boxes.append(norm_box)
            
        return {
            'img_id': item['img_id'],
            'path': item['path'],
            'file_name': item['file_name'],
            'image': img_tensor,
            'label': torch.tensor(item['has_dust_devil'], dtype=torch.float32),
            'bboxes': torch.tensor(normalized_boxes, dtype=torch.float32) if normalized_boxes else torch.zeros((0, 4), dtype=torch.float32),
            'orig_size': (orig_w, orig_h)
        }

def get_dataloader(data_dir, split='train', batch_size=16, shuffle=True, img_size=(416, 416)):
    dataset = MarsDustDevilDataset(data_dir=data_dir, split=split, img_size=img_size)
    
    def collate_fn(batch):
        images = torch.stack([b['image'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        paths = [b['path'] for b in batch]
        file_names = [b['file_name'] for b in batch]
        bboxes = [b['bboxes'] for b in batch]
        orig_sizes = [b['orig_size'] for b in batch]
        return {
            'images': images,
            'labels': labels,
            'paths': paths,
            'file_names': file_names,
            'bboxes': bboxes,
            'orig_sizes': orig_sizes
        }
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataset, loader

if __name__ == '__main__':
    data_dir = r'C:\Users\harsh\Desktop\Martian Dust Devil Detection\data'
    for split in ['train', 'valid', 'test']:
        ds, loader = get_dataloader(data_dir, split=split, batch_size=8)
        print(f"Split {split}: {len(ds)} images loaded.")
