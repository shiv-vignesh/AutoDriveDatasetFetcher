import random
import os
import json
from collections import defaultdict
from typing import List, Dict, Iterable

import numpy as np

import torch
from torch.utils.data import Dataset

import cv2
import albumentations

import transformers
from transformers import BertTokenizer, ViTImageProcessor

from .enums import Enums

class NuScenesObjectDetectDataset(Dataset):    
    
    def __init__(self, table_blob_paths:list, global_captions_paths:list,
                root_dir:str):

        self.sample_tokens = []
        self.tables = {}
        self.global_captions = {}

        self.table_blob_paths = table_blob_paths
        self.global_captions_paths = global_captions_paths

        self.root_dir = root_dir

        if global_captions_paths is not None:        
            for blob_table_path, global_captions_path in zip(table_blob_paths, global_captions_paths):
                self.parse_blob_table(blob_table_path)
                self.parse_global_captions(global_captions_path)    
        else:
            for blob_table_path in table_blob_paths:
                self.parse_blob_table(blob_table_path)

    def parse_global_captions(self, global_captions_path:str):
        global_captions = json.load(open(global_captions_path))
        self.global_captions.update(global_captions)
            
    def parse_blob_table(self, blob_table_path:str):

        table = json.load(open(blob_table_path))
        self.sample_tokens.extend([sample_token for sample_token in table])
        self.tables.update(table)        

    def __len__(self):
        return len(self.sample_tokens)
    
    def __getitem__(self, idx):
        
        sample_token = self.sample_tokens[idx]
        lidar_top_fp = self.tables[sample_token]['lidar_top_fp']
        cam_front_fp = self.tables[sample_token]['cam_front_fp']
        labels_2d_cam_front = self.tables[sample_token]['labels_2d_cam_front']
        
        if self.global_captions_paths is not None and sample_token in self.global_captions:
            captions = self.global_captions[sample_token]
        else:
            captions = None
        
        # splitting because 
        # ../data/nuscenes/trainval04_blobs_US/samples/CAM_FRONT/filename.jpg
        lidar_top_fp = f"{self.root_dir}/{'/'.join(lidar_top_fp.split('/')[3:])}"
        cam_front_fp = f"{self.root_dir}/{'/'.join(cam_front_fp.split('/')[3:])}"
        
        return {
            'sample_token':sample_token, 
            'lidar_top_fp':lidar_top_fp, 
            'cam_front_fp':cam_front_fp, 
            'labels_2d_cam_front':labels_2d_cam_front,
            "captions":captions
        }


"""
----------------------------------------------------------------------------
| ViT Model Variant       | Default Image Size | Patch Size | # of Patches |
| ----------------------- | ------------------ | ---------- | ------------ |
| `vit-base-patch16-224`  | 224 × 224          | 16 × 16    | 196          |
| `vit-large-patch16-384` | 384 × 384          | 16 × 16    | 576          |
| `vit-huge-patch14-224`  | 224 × 224          | 14 × 14    | 256          |
----------------------------------------------------------------------------
"""

class NuScenesRegionCLIP:    
    def __init__(self, image_preprocessor:str="vit_image_preprocessor", 
                text_preprocessor:str="bert_tokenizer", 
                image_resize:tuple=(224, 224), 
                original_size:tuple=(900, 1600),
                positive_mask_type:str="absolute"):
        
        self.image_preprocessor_type = image_preprocessor
        self.text_preprocessor_type = text_preprocessor
        self.image_resize = image_resize
        self.original_size = original_size
        
        self.positive_mask_type = positive_mask_type
        
        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]
        
        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]        
        
        if self.image_preprocessor_type == "vit_image_preprocessor":
            self.image_preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.patch_size = (16, 16)
            self.num_patches = (self.resized_width//self.patch_size[0]) ** 2
        
        if self.text_preprocessor_type == "bert_tokenizer":
            self.text_preprocessor = BertTokenizer.from_pretrained("bert-base-uncased")

        self.transformation = albumentations.Compose(
            [albumentations.LongestMaxSize(max_size=max(self.image_resize)),
            albumentations.PadIfNeeded(
                min_height=self.resized_height,
                min_width=self.resized_width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ), 
            albumentations.CenterCrop(height=self.resized_height, width=self.resized_width, always_apply=True)
            ],                
            bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_labels'])
        )

        self.image_only_transformation = albumentations.Compose(
            [albumentations.Resize(height=self.resized_height, width=self.resized_width, always_apply=True)]
        )

    def preprocess_labels(self, labels_2d_cam_front:Iterable[dict]):

        class_labels = []
        bboxes_2d = []
        
        for label_data in labels_2d_cam_front:

            attribute = label_data['category_name']
            if attribute in Enums.NUSCENES_TO_GENERAL_CLASSES:
                if attribute not in Enums.NUSCENES_TO_GENERAL_CLASSES:
                    continue

                bbox_corners = label_data['bbox_corners']

                left = float(bbox_corners[0])  # left
                top = float(bbox_corners[1])  # top
                right = float(bbox_corners[2])  # right
                bottom = float(bbox_corners[3])  # bottom
                
                class_label = Enums.NUSCENES_TO_GENERAL_CLASSES[attribute]                
                class_labels.append(class_label)
                
                bboxes_2d.append([left, top, right, bottom])

        return class_labels, bboxes_2d
    
    def transform_sample(self, image:np.array, label_bboxes:np.array=None, class_labels:np.array=None):                        

        if label_bboxes is not None and class_labels is not None:        
            transformed_dict = self.transformation(
                image=image, bboxes=label_bboxes, class_labels=class_labels
            )
        else:
            transformed_dict = self.image_only_transformation(
                image=image
            )
        return transformed_dict    
            
    def jitter_bbox(self, bbox:list, scale_factor: float = 0.1, 
                translate_factor: float = 0.3,
                min_visibility: float = 0.5):
        
        """
        scale_factor (float): Max percentage to scale the box's width and height.
        translate_factor (float): Max percentage to translate the box's center.
        min_visibility (float): The minimum fraction of the original box that must
                                be visible in the jittered box.        
        
        1. Apply Random Scaling. Modify height and width 
            + or - factor to create a new height and width.
            
        2. Apply Translation
            Change the centroid of the bbox 
        
        3. Create new bbox and clamp to coordinates        
        """
        
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        h = y_max - y_min
        cx = x_min + w/2
        cy = y_min + h/2
        
        # Apply Random Scaling. Modify height and width 
        new_w_scale = 1.0 + np.random.uniform(-scale_factor, scale_factor)
        new_h_scale = 1.0 + np.random.uniform(-scale_factor, scale_factor)        
        new_w = w * new_w_scale
        new_h = h * new_h_scale
        
        # Apply Translation
        max_dx = w * translate_factor
        max_dy = h * translate_factor
        dx = np.random.uniform(-max_dx, max_dx)
        dy = np.random.uniform(-max_dy, max_dy)
        new_cx = cx + dx
        new_cy = cy + dy
        
        # Convert back to corner coordinates
        new_x_min = new_cx - new_w / 2
        new_y_min = new_cy - new_h / 2
        new_x_max = new_cx + new_w / 2
        new_y_max = new_cy + new_h / 2                
                
        # 4. Clamp to image boundaries
        new_x_min = max(0, new_x_min)
        new_y_min = max(0, new_y_min)
        new_x_max = min(self.resized_width, new_x_max)
        new_y_max = min(self.resized_height, new_y_max)
        
        inter_x_min = max(x_min, new_x_min)
        inter_y_min = max(y_min, new_y_min)
        inter_x_max = min(x_max, new_x_max)
        inter_y_max = min(y_max, new_y_max)
        
        original_area = w*h
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)        
        visibility = inter_area / original_area if original_area > 0 else 0
        
        print(visibility)
        
        if visibility >= min_visibility:
            return (new_x_min, new_y_min, new_x_max, new_y_max)
        else:
            return False        
    
    def create_positive_mask(self, bbox:list, iou_thresh:float=0.1):
        
        H, W = self.image_resize
        H_, W_ = H//self.patch_size[0], W//self.patch_size[1]
        
        mask = torch.zeros((H_, W_), dtype=torch.float16)
    
        x0, y0, x1, y1 = bbox
        # Clamp boxes within image boundaries
        x0, x1 = max(0, x0), min(W - 1, x1)
        y0, y1 = max(0, y0), min(H - 1, y1)
        
        pW, pH = self.patch_size
        
        i_start, i_end = int(y0//pH), int(y1//pH)
        j_start, j_end = int(x0//pW), int(x1//pW)
        
        if self.positive_mask_type == "absolute":        
            for i in range(i_start, i_end+1):
                for j in range(j_start, j_end+1):
                    mask[i, j] = 1
        
        elif self.positive_mask_type == "weighted":
            
            patch_area = float(pW * pH)
            for i in range(i_start, i_end+1):
                for j in range(j_start, j_end+1):                    
                    patch_x0, patch_y0 = j * pW, i * pH
                    patch_x1, patch_y1 = patch_x0 + pW, patch_y0 + pH
                    # Calculate intersection area
                    inter_x0 = max(x0, patch_x0)
                    inter_y0 = max(y0, patch_y0)
                    inter_x1 = min(x1, patch_x1)
                    inter_y1 = min(y1, patch_y1)

                    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
                    overlap = inter_area / patch_area  # soft weight
                    
                    mask[i, j] = overlap

        return mask.flatten()
    
    def preprocess(self, data_items:dict):        
        """
        1. Read image from cv2.imread(cam_front_fp)
        2. Resize the image to image size. Use albumentations. 
            - transformed_dict['image'], transformed_dict['labels']
        3. Use Query Templates to transform
            - labels to <Query> --> labels list
            - Tokenize labels list to 
                - tensor shape (total_labels, max_seq_len)
                - tensor shape (total_labels, batch_idx)

        4. Perform region matching 
            - patch idx and label using bbox info. 
            - create positive masks, 
                - soft negative masks - different classes, no root class match. 
                - hard negative masks - same root class, but different class. 
                
        5. Return 
            - (B, N) = image patches tensor (B, 3, 224, 224)
            - (total_labels, max_seq_len) = text tensor 
            - (total_labels, max_seq_len) = text attn mask
            - positive masks 
            - soft negative masks
            - hard negative masks             
        """        
        
        
        batch_images = []
        batch_labels = [] #(total_samples)
        batch_bbox = [] #(total_samples, 4)
        sample_tokens = []
        
        batch_preprocessed_items = {}

        for data_item in data_items:
            sample_token = data_item['sample_token']
            cam_front_fp = data_item['cam_front_fp']
            labels_2d_cam_front = data_item['labels_2d_cam_front']
            
            cam_front_image = cv2.imread(cam_front_fp)
            
            class_labels, label_bboxes_2d = self.preprocess_labels(labels_2d_cam_front)
            
            transformed_dict = self.transform_sample(
                cam_front_image, label_bboxes_2d, class_labels
            )

            # cam_front_image = transformed_dict['image']
            class_labels, label_bboxes_2d = transformed_dict['class_labels'], transformed_dict['bboxes']
            
            for class_label, bboxes in zip(class_labels, label_bboxes_2d):
                
                label_mask = self.create_positive_mask(bboxes)
                print(f'Original {class_label} {label_mask[label_mask == 1].sum()}')
                
                jitter_bbox = self.jitter_bbox(bboxes)
                label_mask = self.create_positive_mask(jitter_bbox)
                print(f'Jitter {class_label} {label_mask[label_mask == 1].sum()}')    
                
                print()            
                
            # exit(1)
            
            if self.image_preprocessor_type == "vit_image_preprocessor":
                pixel_values = self.image_preprocessor(cam_front_image, return_tensors="pt").pixel_values

            if self.text_preprocessor_type == "bert_tokenizer":
                pass
            
        exit(1)

    
    def __call__(self, data_items:dict):
        return self.preprocess(data_items)
    
class NuScenesCLIPCollateFn:
    def __init__(self, image_preprocessor:str="vit_image_preprocessor", 
                text_preprocessor:str="bert_tokenizer", 
                image_resize:tuple=(224, 224), 
                original_size:tuple=(900, 1600)):
    
        self.image_preprocessor_type = image_preprocessor
        self.text_preprocessor_type = text_preprocessor
        self.image_resize = image_resize
        self.original_size = original_size
        
        self.original_width = self.original_size[1]
        self.original_height = self.original_size[0]

        self.resized_width = self.image_resize[1]
        self.resized_height = self.image_resize[0]         

        if self.image_preprocessor_type == "vit_image_preprocessor":
            self.image_preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.patch_size = (16, 16)
            self.num_patches = (self.resized_width//self.patch_size[0]) ** 2

        if self.text_preprocessor_type == "bert_tokenizer":
            self.text_preprocessor = BertTokenizer.from_pretrained("bert-base-uncased")   
   
        self.image_only_transformation = albumentations.Compose(
            [albumentations.Resize(height=self.resized_height, width=self.resized_width, always_apply=True)]
        )

    def create_count_map(self, labels_2d_cam_front):

        count_map = defaultdict(int)
        for obj in labels_2d_cam_front:
            category = obj["category_name"]
            count_map[category] += 1

        return count_map
    
    def create_similarity_matrix(self, batch_object_counts:list):
        
        batch_size = len(batch_object_counts)
        sim_matrix = torch.zeros((batch_size, batch_size))
        
        for i in range(batch_size):
            set_i = set(batch_object_counts[i].keys())
            for j in range(batch_size):
                set_j = set(batch_object_counts[j].keys())
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                sim_matrix[i, j] = intersection / union if union > 0 else 0.0

        return sim_matrix        
    
    def preprocess(self, data_items:dict):
        """
        1. Read image from cv2.imread(cam_front_fp)
        2. Resize the image to image size. Use albumentations. 
            - transformed_dict['image']
        3. Use Query Templates to transform
            - labels to <Query> --> labels list
            - Tokenize labels list to 
                - tensor shape (total_labels, max_seq_len)
                - tensor shape (total_labels, batch_idx)

        4. Return 
            - (B, N) = image patches tensor (B, 3, 224, 224)
            - (B, N) = captions tensor (B, max_seq_len)            
        """

        batch_pixel_values = []
        batch_captions = []
        sample_tokens = []

        batch_object_counts = []

        for data_item in data_items:
            
            if data_item['captions'] is None:
                continue
            
            cam_front_fp = data_item['cam_front_fp']
            captions = data_item['captions']['llava_generated_captions']

            cam_front_image = cv2.imread(cam_front_fp)

            if self.image_preprocessor_type == "vit_image_preprocessor":
                pixel_values = self.image_preprocessor(cam_front_image, return_tensors="pt").pixel_values
                batch_pixel_values.append(pixel_values.squeeze(0))

            generated_caption = random.choice(captions)
            batch_captions.append(generated_caption)
            
            count_map = self.create_count_map(data_item['labels_2d_cam_front'])
            batch_object_counts.append(count_map)

        if self.text_preprocessor_type == "bert_tokenizer":
            encoded_inputs = self.text_preprocessor(batch_captions, padding=True, truncation=True, return_tensors="pt")
            batch_input_ids = encoded_inputs['input_ids']
            batch_attention_masks = encoded_inputs['attention_mask']

        batch_pixel_values = torch.stack(batch_pixel_values, dim=0)
        sim_matrix = self.create_similarity_matrix(batch_object_counts)

        return {
            "batch_pixel_values":batch_pixel_values, 
            "batch_input_ids":batch_input_ids,
            "batch_attention_masks":batch_attention_masks,
            "sim_matrix":sim_matrix
        }

    def __call__(self, data_items:dict):
        return self.preprocess(data_items)
    
if __name__ == "__main__":
    
    dataset =NuScenesObjectDetectDataset(
        table_blob_paths=['../trainval03_blobs_US/tables.json'],
        global_captions_paths=['../trainval03_blobs_US/global_captions.json'],
        root_dir='../'
    )
    
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=4, 
        collate_fn=NuScenesCLIPCollateFn()
    )
    
    for data in dataloader:
        for k, v in data.items():
            if torch.is_tensor(v):
                print(f'{k} {v.shape}')
                
        exit(1)