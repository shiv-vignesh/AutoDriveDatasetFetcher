import os, json
import torch
import faiss

from dataset_utils.retriever import ImageBindRetriever
from dataset_utils.nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset\
    
from dataset_utils.evaluator import recall_k, plot_similarity_score_hist

def _create_device(device:torch):    
    return torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

def build_embeddings(retriever_kwargs:dict):
    
    device = retriever_kwargs['retriever_kwargs']['ImageBind_kwargs']['device']
    output_dir = retriever_kwargs['retriever_kwargs']['embeddings_dir']
    
    if not os.path.exists(f'{output_dir}/embeddings'):
        os.makedirs(f'{output_dir}/embeddings')

    image_bind_retriver = ImageBindRetriever(device=device)

    nuscenes_dataset_kwargs = retriever_kwargs['dataset_kwargs']['nuscenes_dataset_kwargs']
    
    dataset = NuScenesObjectDetectDataset(
        table_blob_paths=nuscenes_dataset_kwargs['table_blob_paths'],
        root_dir=nuscenes_dataset_kwargs['root_dir']
    )
    
    index, token_id_to_image_path, faiss_index_id_to_token = image_bind_retriver._create_embeddings(dataset, f'{output_dir}/embeddings')
    
    faiss.write_index(index, f'{output_dir}/faiss_{type(index)}.faiss')
    
    with open(f'{output_dir}/token_id_to_image_path.json', 'w+') as f:
        json.dump(token_id_to_image_path, f)
        
    with open(f'{output_dir}/faiss_index_id_to_token.json', 'w+') as f:
        json.dump(faiss_index_id_to_token, f)        

if __name__ == "__main__":

    retriever_kwargs = json.load(open(f'config/retriever_config.json'))
    
    build_embeddings(retriever_kwargs)