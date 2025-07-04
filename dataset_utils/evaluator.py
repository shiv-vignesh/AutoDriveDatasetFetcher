
from functools import reduce
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

from .nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset

def plot_similarity_score_hist(similarity_scores: list[float], save_path:str, bins: int = 50):
    """
    Plots a histogram of similarity scores.

    Arguments:
    - similarity_scores: list of cosine similarities (float values)
    - bins: number of histogram bins
    """
    plt.figure(figsize=(8, 5))
    plt.hist(similarity_scores, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Similarity Score Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{save_path}/similarity_score_hist.png')

def plot_common_object_histogram_per_category(per_category_stats, save_path:str):
    categories = list(per_category_stats.keys())
    common_counts = [per_category_stats[cat]["retrieved"] for cat in categories]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(categories, common_counts, color='cornflowerblue')
    plt.xlabel("Object Category")
    plt.ylabel("Retrieved with Correct Co-occurring Objects")
    plt.title("Number of Common Object Types Retrieved (Per Category)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{int(height)}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
        
    plt.savefig(f'{save_path}/common_object_histogram.png')

def recall_k(retrieved_results: dict, save_path:str):
    """
    Computes Recall@k: how often the ground truth sample's label string appears in the top-k retrieved tokens.
    
    Arguments:
    - retrieved_results: dict of sample_token → {'retrieved_tokens': list of sample_tokens}
    - ground_truth: dict of sample_token → label_str (used to match retrieved sample content)
    - k: how many retrieved samples to consider

    Returns:
    - recall_k_score: float [0, 1]
    """
    correct = 0
    total = 0
    
    for sample_token, retrieved in retrieved_results.items():
        
        match_found = False
        for retrieved_token in retrieved:
            if retrieved_token == sample_token:
                match_found = True
                break
        
        correct += int(match_found)
        total += 1

    recall_k_score = correct / total if total > 0 else 0.0
    
    with open(f'{save_path}/recall_k.txt', 'w+') as f:
        f.write(f'Correct: {correct} Total: {total} Recall_k_score: {recall_k_score}')

def analyze_common_objects_and_count_match(retrieved_results: dict, dataset:NuScenesObjectDetectDataset):
    
    def _object_stats(ground_truth_labels:dict, retrieved_truth_labels:dict):

        common_object_stats = []
        count_match_scores = []
        
        common_objects = set(ground_truth_labels.keys()) & set(retrieved_truth_labels.keys())
        common_object_stats.append(len(common_objects))
        
        if common_objects:
            match_count = sum(1 for obj in common_objects if ground_truth_labels[obj] == ground_truth_labels[obj])
            match_score = match_count / len(common_objects)
            count_match_scores.append(match_score)
        else:
            count_match_scores.append(0.0)
            
        return common_object_stats, count_match_scores

    category_match_stats = defaultdict(lambda: {
        "appearances": 0,       # how often the category appears in the query
        "retrieved": 0,         # how often retrieved sample contains the category
        "exact_match_count": 0, # how often count matches
        "match_ratios": []      # per retrieval: [0.0 to 1.0]
    })
    
    global_common_object_stats = []
    global_count_match_scores = []    
        
    for sample_token in tqdm(retrieved_results):
        retrieved_tokens = retrieved_results[sample_token]['retrieved_tokens']
        ground_truth_labels = dataset.tables[sample_token]['labels_2d_cam_front']
        ground_truth_labels = [label_dict['category_name'] for label_dict in ground_truth_labels]

        ground_truth_labels_dict = reduce(
            lambda acc, x: {**acc, x: acc.get(x, 0) + 1},
            ground_truth_labels,
            {}
        )

        for category in ground_truth_labels_dict:
            category_match_stats[category]["appearances"] += 1
        
        for retrieved_token in retrieved_tokens:
            retrieved_truth_labels = dataset.tables[retrieved_token]['labels_2d_cam_front']
            retrieved_truth_labels = [label_dict['category_name'] for label_dict in retrieved_truth_labels]
        
            retrieved_truth_labels_dict = reduce(
                lambda acc, x: {**acc, x: acc.get(x, 0) + 1},
                retrieved_truth_labels,
                {}
            )
            
            common_object_stats, count_match_scores = _object_stats(
                ground_truth_labels_dict, retrieved_truth_labels_dict
            )

            global_common_object_stats.extend(common_object_stats)
            global_count_match_scores.extend(count_match_scores)

            # Per-category stats
            for category in ground_truth_labels_dict:
                if category in retrieved_truth_labels_dict:
                    category_match_stats[category]["retrieved"] += 1

                    # Exact match check
                    if ground_truth_labels_dict[category] == retrieved_truth_labels_dict[category]:
                        category_match_stats[category]["exact_match_count"] += 1

                    # Partial match ratio
                    gt_count = ground_truth_labels_dict[category]
                    rt_count = retrieved_truth_labels_dict[category]
                    match_ratio = min(gt_count, rt_count) / max(gt_count, rt_count)
                    category_match_stats[category]["match_ratios"].append(match_ratio)

    return {
        "global_common_object_stats": global_common_object_stats,
        "global_count_match_scores": global_count_match_scores,
        "per_category_stats": category_match_stats
    }