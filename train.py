import os
import torch

from models.model import CLIP
from trainer.trainer import Trainer

def load_model(model_kwargs:dict):
    
    text_model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vision_model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    clip = CLIP(model_kwargs['text_model_name'],
                model_kwargs['vision_model_name'],
                text_model_device, vision_model_device)
    
    return clip

def train(dataset_kwargs:dict, model_kwargs:dict, optimizer_kwargs:dict,
            lr_scheduler_kwargs:dict, trainer_kwargs:dict):
    
    clip = load_model(model_kwargs)
    
    trainer = Trainer(
        model=clip,
        dataset_kwargs=dataset_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        trainer_kwargs=trainer_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs
    )
    
    trainer.train()
    

if __name__ == "__main__":
    
    dataset_kwargs = {
        "train_dataset_kwargs" : {
            "table_blob_paths":['data/trainval03_blobs_US/tables.json', 'data/trainval04_blobs_US/tables.json'],
            "global_captions_paths":['data/trainval03_blobs_US/global_captions.json', 'data/trainval04_blobs_US/global_captions.json'],
            "root_dir":'data/',
            "batch_size":24,
            "shuffle":True
        },
        "eval_dataset_kwargs" : {
            "table_blob_paths":['data/trainval03_blobs_US/tables.json', 'data/trainval04_blobs_US/tables.json'],
            "global_captions_paths":['data/trainval03_blobs_US/global_captions.json', 'data/trainval04_blobs_US/global_captions.json'],
            "root_dir":'data/',
            "batch_size":24,
            "shuffle":False
        }            
    }
    
    model_kwargs = {
        "text_model_name":"bert_model",
        "vision_model_name":"vit_model",
        "text_model_device":"cpu",
        "vision_model_device":"cpu",
        "model_path":""
    }
    
    optimizer_kwargs = {
        "_type":"AdamW",
        "text_model_lr": 5e-5,
        "vision_model_lr": 5e-5,
        "text_proj_lr": 1e-3,
        "vision_proj_lr": 1e-3,
        "momentum":0.9,
        "weight_decay":0.01
    }
    
    lr_scheduler_kwargs = {
        "_type":"cosine",
        "linear_lr_kwargs":{
            "start_factor":1.0,
            "end_factor":0.01
        },
        "cosine_annealing_lr_kwargs":{
            "eta_min":1e-5
        }
    }    
    
    trainer_kwargs = {
        'output_dir':"training_results/GlobalCaptionsClipTraining",
        "gradient_clipping":2.0,
        'checkpoint_idx':1,
        "epochs":50,
        "gradient_accumulation_steps":2
    }
    
    train(
        dataset_kwargs,
        model_kwargs,
        optimizer_kwargs, 
        lr_scheduler_kwargs, 
        trainer_kwargs
    )
    
