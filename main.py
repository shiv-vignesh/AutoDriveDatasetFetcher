import torch
import faiss
import json, os

from dataset_utils.retriever import ImageBindRetriever
from dataset_utils.nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset

from dataset_utils.evaluator import recall_k, plot_similarity_score_hist, analyze_common_objects_and_count_match, plot_common_object_histogram_per_category

def retrieve(faiss_index_path:str, faiss_index_id_to_token:str, token_id_to_image_path:str, save_path:str):
    
    nuscenes_dataset_kwargs = retriever_kwargs['dataset_kwargs']['nuscenes_dataset_kwargs']
    dataset = NuScenesObjectDetectDataset(
        table_blob_paths=nuscenes_dataset_kwargs['table_blob_paths'],
        root_dir=nuscenes_dataset_kwargs['root_dir']
    )
    
    if not os.path.exists(f'{save_path}/retrieved_results.json'):

        index = faiss.read_index(faiss_index_path)

        device = retriever_kwargs['retriever_kwargs']['ImageBind_kwargs']['device']        
        image_bind_retriver = ImageBindRetriever(device=device)        
        
        retrieved_results, similarity_scores = image_bind_retriver(
            dataset, index, 
            faiss_index_id_to_token, 
            token_id_to_image_path, 
            topk=1
        )
        
        recall_k(retrieved_results, save_path)
        plot_similarity_score_hist(similarity_scores, save_path)
        
        with open(f'{save_path}/retrieved_results.json','w+') as f:
            json.dump(retrieved_results, f)
            
    else:
        retrieved_results = json.load(open(f'{save_path}/retrieved_results.json'))
        
    results = analyze_common_objects_and_count_match(
        retrieved_results, dataset
    )
    
    plot_common_object_histogram_per_category(
        results['per_category_stats'], f'{save_path}'
    )

if __name__ == "__main__":
    
    retriever_kwargs = json.load(open(f'config/retriever_config.json'))
    faiss_index_id_to_token = json.load(open(retriever_kwargs['retriever_kwargs']['faiss_index_id_to_token']))
    token_id_to_image_path = json.load(open(retriever_kwargs['retriever_kwargs']['token_id_to_image_path']))
    
    if not os.path.exists(retriever_kwargs['retriever_kwargs']['output_path']):
        os.makedirs(retriever_kwargs['retriever_kwargs']['output_path'])
    
    retrieve(
        retriever_kwargs['retriever_kwargs']['faiss_index_path'],
        faiss_index_id_to_token, token_id_to_image_path, save_path=retriever_kwargs['retriever_kwargs']['output_path']
    )
