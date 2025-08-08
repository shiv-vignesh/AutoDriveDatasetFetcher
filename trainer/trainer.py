import os, time
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .logger import Logger
from dataset_utils.nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset, NuScenesCLIPCollateFn
from models.model import CLIP

class Trainer:
    
    def __init__(self, model:CLIP, dataset_kwargs:dict, optimizer_kwargs:dict,
                trainer_kwargs:dict, lr_scheduler_kwargs:dict):

        self.model = model

        self.output_dir = trainer_kwargs['output_dir']
        self.gradient_clipping = trainer_kwargs["gradient_clipping"]
        self.checkpoint_idx = trainer_kwargs['checkpoint_idx']
        self.epochs = trainer_kwargs["epochs"]
        self.gradient_accumulation_steps = trainer_kwargs['gradient_accumulation_steps']        
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)                
        
        self.logger = Logger(self.output_dir)

        self._init_dataloader(dataset_kwargs)

        self.total_train_batch = len(self.train_dataloader)
        self.ten_percent_train_batch = self.total_train_batch // 10
        
        if type(self.train_dataloader.dataset) == NuScenesObjectDetectDataset:
            self.logger.log_message(f'  Training on NuScenes Dataset   ')
            self.logger.log_message(f'  Blobs Files: {self.train_dataloader.dataset.table_blob_paths}')
            self.logger.log_message(f'Train Batch Size: {self.train_dataloader.batch_size}')
            
            self.logger.log_new_line()
            
        if type(self.validation_dataloader.dataset) == NuScenesObjectDetectDataset:
            self.logger.log_message(f'  Training on NuScenes Dataset   ')
            self.logger.log_message(f'  Blobs Files: {self.validation_dataloader.dataset.table_blob_paths}')
            self.logger.log_message(f'Train Batch Size: {self.validation_dataloader.batch_size}')            
        
        self._init_optimizer(optimizer_kwargs)
        self.logger.log_line()
        self.logger.log_message(f'Optimizer: {self.optimizer.__class__.__name__}')
        self.logger.log_new_line()
        
        if lr_scheduler_kwargs:
            self._init_lr_scheduler(lr_scheduler_kwargs)
        
        self.logger.log_line()
        self.logger.log_message(f'Text Model {self.model.text_model_name} -- Device: {self.model.text_model_device}')
        self.logger.log_message(f'Vision Model {self.model.vision_model_name} -- Device: {self.model.vision_model_device}')
        self.logger.log_new_line()
        
    def _init_dataloader(self, dataset_kwargs:dict):

        def _create_dataloader(dataset_kwargs:dict):
            
            dataset = NuScenesObjectDetectDataset(
                dataset_kwargs['table_blob_paths'],
                dataset_kwargs['global_captions_paths'],
                root_dir=dataset_kwargs['root_dir']
            )
            
            return DataLoader(
                dataset=dataset,
                batch_size=dataset_kwargs['batch_size'],
                shuffle=dataset_kwargs['shuffle'],
                collate_fn=NuScenesCLIPCollateFn()
            )
        
        self.train_dataloader = _create_dataloader(dataset_kwargs['train_dataset_kwargs'])
        self.validation_dataloader = _create_dataloader(dataset_kwargs['eval_dataset_kwargs'])
        
    def _init_optimizer(self, optimizer_kwargs:dict):
        
        params_dict = []

        params_dict.append({
            'params':self.model.text_model.parameters(), 
            'lr':optimizer_kwargs['text_model_lr'], 
            'model_name':f'TextBackbone_{self.model.text_model.__class__.__name__}'
        })
        
        params_dict.append({
            'params':self.model.text_projection.parameters(), 
            'lr':optimizer_kwargs['text_proj_lr'], 
            'model_name':f'TextBackbone_{self.model.text_model.__class__.__name__}'
        })        

        params_dict.append({
            'params':self.model.vision_model.parameters(), 
            'lr':optimizer_kwargs['vision_model_lr'], 
            'model_name':f'VisionBackbone_{self.model.vision_model.__class__.__name__}'
        })
        
        params_dict.append({
            'params':self.model.visual_projection.parameters(), 
            'lr':optimizer_kwargs['vision_proj_lr'], 
            'model_name':f'VisionBackbone_{self.model.vision_model.__class__.__name__}'
        })                
        
        if optimizer_kwargs['_type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                params_dict,
                weight_decay=optimizer_kwargs['momentum']
            )
            
    def _init_lr_scheduler(self, lr_scheduler_kwargs:dict):

        if lr_scheduler_kwargs['_type'] == "linear":
            lr_scheduler_kwargs = lr_scheduler_kwargs['linear_lr_kwargs']            
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=lr_scheduler_kwargs['start_factor'],
                end_factor=lr_scheduler_kwargs['end_factor'],
                total_iters=self.epochs
            )

            self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
            self.logger.log_message(f'LR Scheduler Start Factor: {lr_scheduler_kwargs["start_factor"]}')
            self.logger.log_message(f'LR Scheduler End Factor: {lr_scheduler_kwargs["end_factor"]}')
            self.logger.log_new_line()            

        elif lr_scheduler_kwargs['_type'] == "cosine":
            lr_scheduler_kwargs = lr_scheduler_kwargs['cosine_annealing_lr_kwargs']            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=lr_scheduler_kwargs['eta_min']
            )

            self.logger.log_message(f'LR Scheduler: {self.lr_scheduler.__class__.__name__}')
            self.logger.log_message(f'LR Scheduler TMax: {self.epochs}')
            self.logger.log_message(f'LR Scheduler ETA Min: {lr_scheduler_kwargs["eta_min"]}')            
            self.logger.log_new_line()
            
    def train(self):
        
        self.logger.log_line()
        self.logger.log_message(
            f'Training: Max Epoch - {self.epochs}'
        )
        self.logger.log_new_line()

        self.total_training_time = 0.0
        self.cur_epoch = 0
        
        for epoch in range(1, self.epochs+1):
            self.cur_epoch = epoch
            self.logger.log_line()
            
            self.train_one_epoch()
            self.valid_one_epoch()

            torch.save(
                self.model.state_dict(), f'{self.output_dir}/ckpt-model.pt'
            )

    def train_one_epoch(self):
        
        self.model.train()
        
        total_loss = 0.0 
        total_loss = 0.0 
        ten_percent_batch_total_loss = 0
        
        epoch_training_time = 0.0
        ten_percent_training_time = 0.0
        ten_percent_metric_per_grid = defaultdict(lambda:defaultdict(int))
        
        train_iter = tqdm(self.train_dataloader, desc=f'Training Epoch: {self.cur_epoch}')
        for batch_idx, data_items in enumerate(train_iter):
            
            step_begin_time = time.time()
            loss, outputs = self.train_one_step(data_items)            
            step_end_time = time.time()

            if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx == self.train_dataloader.__len__() - 1):
                
                self.optimizer.step()
                self.lr_scheduler.step()

                self.optimizer.zero_grad()                                            
                current_lr = self.optimizer.param_groups[0]['lr']
                
            total_loss += loss.item()
            ten_percent_batch_total_loss += loss.item()

            epoch_training_time += (step_end_time - step_begin_time)
            ten_percent_training_time += (step_end_time - step_begin_time)
            
            if (batch_idx + 1) % self.ten_percent_train_batch == 0:
                average_loss = ten_percent_batch_total_loss/self.ten_percent_train_batch
                average_time = ten_percent_training_time/self.ten_percent_train_batch    

                message = f'Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - total loss {average_loss:.4f} -- current_lr: {current_lr}'
                self.logger.log_message(message=message)
                self.logger.log_new_line()                

                ten_percent_batch_total_loss = 0
                ten_percent_training_time = 0.0
                ten_percent_metric_per_grid = defaultdict(lambda:defaultdict(int))
                
        self.logger.log_message(
            f'Epoch {self.cur_epoch} - Average Loss {total_loss/self.total_train_batch:.4f} -- current_lr: {current_lr}'
        )            
            
    def train_one_step(self, data_items):

        with torch.set_grad_enabled(True):
            outputs = self.model(
                data_items['batch_pixel_values'],
                data_items['batch_input_ids'],
                data_items['batch_attention_masks']
            )
            
            loss = outputs['loss']
            
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)

        return loss, outputs
    
    
    def valid_one_epoch(self):
        self.model.eval()

        total_loss = 0.0
        metric_accumulator = defaultdict(float)
        num_batches = len(self.validation_dataloader)

        val_iter = tqdm(self.validation_dataloader, desc=f'Validation Epoch: {self.cur_epoch}')
        all_image_features = []
        all_text_features = []
        
        for batch in val_iter:
            with torch.no_grad():
                outputs = self.model(
                    batch["batch_pixel_values"],
                    batch["batch_input_ids"],
                    batch["batch_attention_masks"]
                )

            # Aggregate loss
            total_loss += outputs["loss"].item()

            # Compute retrieval metrics for this batch
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logit_scale = self.model.logit_scale.exp().detach().cpu()

            batch_metrics = self.compute_clip_retrieval_metrics(image_features.detach().cpu(), text_features.detach().cpu(), logit_scale, prefix="batch_")
            
            all_image_features.append(image_features.detach().cpu())
            all_text_features.append(text_features.detach().cpu())

            for k, v in batch_metrics.items():
                metric_accumulator[k] += v

        # Average metrics across batches
        final_metrics = {k: v / num_batches for k, v in metric_accumulator.items()}
        average_loss = total_loss / num_batches

        self.logger.log_line()
        self.logger.log_message(f'Validation Epoch {self.cur_epoch} - Avg Loss: {average_loss:.4f}')
        for metric_name, value in final_metrics.items():
            self.logger.log_message(f'{metric_name}: {value:.4f}')
        self.logger.log_new_line()

        for k, v in batch_metrics.items():
            metric_accumulator[k] += v

        # Average batch-wise metrics
        batch_metrics_avg = {k: v / num_batches for k, v in metric_accumulator.items()}

        # Compute full-dataset metrics
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        full_metrics = self.compute_clip_retrieval_metrics(
            image_features=all_image_features,
            text_features=all_text_features,
            logit_scale=logit_scale,
            prefix="full_"
        )

        # Final logging
        average_loss = total_loss / num_batches

        self.logger.log_line()
        self.logger.log_message(f'Validation Epoch {self.cur_epoch} - Avg Loss: {average_loss:.4f}')

        for metric_name, value in sorted(batch_metrics_avg.items()):
            self.logger.log_message(f'{metric_name}: {value:.4f}')
        for metric_name, value in sorted(full_metrics.items()):
            self.logger.log_message(f'{metric_name}: {value:.4f}')
            
        self.logger.log_new_line()

    def compute_clip_retrieval_metrics(self, image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor, prefix:str):
        """
        Compute retrieval metrics between image and text features using scaled logits.

        Args:
            image_features (Tensor): shape (batch_size, feature_dim)
            text_features (Tensor): shape (batch_size, feature_dim)
            logit_scale (Tensor): scalar tensor for temperature scaling

        Returns:
            Dict[str, float]: Retrieval metrics
        """
        assert image_features.shape[0] == text_features.shape[0], "Batch size mismatch between image and text features"

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        N = text_features.size(0)
        ground_truth = torch.arange(N).view(-1, 1).to(image_features.device)

        metrics = {}
        for name, logits in {"image_to_text": logits_per_image, "text_to_image": logits_per_text}.items():
            ranking = torch.argsort(logits, dim=-1, descending=True)
            preds = torch.where(ranking == ground_truth)[1].detach().cpu().numpy()

            metrics[f"{prefix}{name}_mean_rank"] = float(np.mean(preds) + 1)
            metrics[f"{prefix}{name}_median_rank"] = float(np.floor(np.median(preds)) + 1)
            for k in [1, 5, 10]:
                metrics[f"{prefix}{name}_R@{k}"] = float(np.mean(preds < k))

        return metrics
