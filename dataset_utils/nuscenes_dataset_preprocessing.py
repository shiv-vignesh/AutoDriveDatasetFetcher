import os
import json
from typing import List, Dict, Iterable

import torch
from torch.utils.data import Dataset

class NuScenesObjectDetectDataset(Dataset):    
    
    def __init__(self, table_blob_paths:list, root_dir:str):
        
        self.sample_tokens = []
        self.tables = {}
        self.table_blob_paths = table_blob_paths
        
        self.root_dir = root_dir
                
        for blob_table_path in table_blob_paths:
            self.parse_blob_table(blob_table_path)
    
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
        
        # splitting because 
        # ../data/nuscenes/trainval04_blobs_US/samples/CAM_FRONT/filename.jpg
        lidar_top_fp = f"{self.root_dir}/{'/'.join(lidar_top_fp.split('/')[3:])}"
        cam_front_fp = f"{self.root_dir}/{'/'.join(cam_front_fp.split('/')[3:])}"
        
        return {
            'sample_token':sample_token, 
            'lidar_top_fp':lidar_top_fp, 
            'cam_front_fp':cam_front_fp, 
            'labels_2d_cam_front':labels_2d_cam_front
        }


