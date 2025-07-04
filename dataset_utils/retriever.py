import os
from tqdm import tqdm
import numpy as np

from collections import defaultdict

import faiss
import torch

from ImageBind.imagebind.models import imagebind_model
from ImageBind.imagebind.models.imagebind_model import ModalityType
from ImageBind.imagebind import data

from dataset_utils.nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset

class ImageBindRetriever:
    
    def __init__(self, device:torch.device):
        
        self.device = device        
        self.model = self._load_model().to(self.device)                
        
    def _load_model(self):    
        return imagebind_model.imagebind_huge(pretrained=True)
    
    def preprocess_labels(self, labels_2d_cam_front:dict):
        attributes = [label_data['category_name'] for label_data in labels_2d_cam_front]
        attributes = ' '.join(attributes)
        return f'This Scene has: {attributes}'
    
    def _create_embeddings(self, dataset:NuScenesObjectDetectDataset, output_dir:str):        
        
        def save_npy_file(embeddings:torch.tensor, file_path:str):
            """
            Saves a torch tensor as a `.npy` file.
            Args:
                embeddings (torch.Tensor): Embeddings to save
                file_path (str): Path to save the file
            """
            embeddings = embeddings.cpu().numpy()
            with open(f'{file_path}', 'wb') as f:
                np.save(f, embeddings)        
        
        self.model.eval()
        token_id_to_image_path = {}
        faiss_index_id_to_token = {}
        
        d = 1024
        index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

        for idx, data_items in enumerate(tqdm(dataset)):
            sample_token = data_items['sample_token']
            image_fp = data_items['cam_front_fp']
            token_id_to_image_path[sample_token] = image_fp
            
            if not os.path.exists(f'{output_dir}/{sample_token}.npy'):
                label_str = self.preprocess_labels(data_items['labels_2d_cam_front'])
                
                _input = {
                    ModalityType.TEXT : data.load_and_transform_text([label_str], self.device),
                    ModalityType.VISION : data.load_and_transform_vision_data([image_fp], self.device)
                }
                
                with torch.no_grad():
                    embeddings = self.model(_input)
                    embed_image = embeddings[ModalityType.VISION].squeeze(0)
                    
                save_npy_file(embed_image, f'{output_dir}/{sample_token}.npy')

                embed_image = embed_image.cpu().numpy()
                embed_image = embed_image.reshape(1, -1)

            else:    
                embed_image = np.load(f'{output_dir}/{sample_token}.npy').astype('float32').reshape(1, -1)
            
            index.add_with_ids(
                embed_image, 
                np.array([idx], dtype='int64')
            )

            faiss_index_id_to_token[idx] = sample_token

        return index, token_id_to_image_path, faiss_index_id_to_token

    def __call__(self, dataset:NuScenesObjectDetectDataset, faiss_index:faiss.Index, 
                faiss_index_id_to_token:dict, token_id_to_image_path:dict, topk:int):
        
        self.model.eval()
        _iter = iter(tqdm(dataset, desc=f'Performing Retrieval based on text prompts'))

        retrieved_results = defaultdict()
        similarity_scores = []
        
        for data_items in tqdm(_iter):
            sample_token = data_items['sample_token']
            image_fp = data_items['cam_front_fp']
            
            label_str = self.preprocess_labels(data_items['labels_2d_cam_front'])
            
            _input = {
                ModalityType.TEXT : data.load_and_transform_text([label_str], self.device),
                ModalityType.VISION : data.load_and_transform_vision_data([image_fp], self.device)
            }
            
            with torch.no_grad():
                embeddings = self.model(_input)
                embed_text = embeddings[ModalityType.TEXT].cpu().numpy()
                embed_image = embeddings[ModalityType.VISION].cpu().numpy()
                                  
            faiss.normalize_L2(embed_text)
            _, embed_indices = faiss_index.search(embed_text, topk)
            
            embed_text = embed_text.squeeze()   # shape: (D,)
            embed_image = embed_image.squeeze() # shape: (D,)

            # Step 2: Normalize
            embed_text = embed_text / np.linalg.norm(embed_text)
            embed_image = embed_image / np.linalg.norm(embed_image)

            # Step 3: Cosine similarity
            similarity_score = np.dot(embed_text, embed_image)  # scalar in [-1, 1]
            
            retrived_tokens = [faiss_index_id_to_token[str(embed_index)] for embed_index in embed_indices.flatten().tolist()]

            retrieved_results[sample_token] = {
                'retrieved_tokens':retrived_tokens, 
                'similarity_score':float(similarity_score)
            }

            similarity_scores.append(similarity_score)
            
        return retrieved_results, similarity_scores