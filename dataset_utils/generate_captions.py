import json
from tqdm import tqdm

import torch 
from collections import defaultdict
from transformers import AutoProcessor, LlavaForConditionalGeneration

from nuscenes_dataset_preprocessing import NuScenesObjectDetectDataset

"""
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": ""},
            {"type": "text", "text": ""},
        ],
    },
]
"""
def load_llava_hf():
    # Load the model in half-precision
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    return model, processor

def create_dataloader(table_blob_paths:str):
    
    dataset = NuScenesObjectDetectDataset(
        table_blob_paths=table_blob_paths,
        root_dir='../'
    )
    
    return dataset

def summarize_scene_objects(labels_2d_cam_front):
    object_summary = defaultdict(lambda: {'count': 0, 'moving': 0, 'standing': 0, 'parked': 0, 'bbox': []})
    
    for obj in labels_2d_cam_front:
        category = obj["category_name"]  # e.g., "vehicle.car"
        attr_tokens = obj.get("attribute_tokens", [])
        bbox = obj.get("bbox_corners", [])

        # Map attributes (NuScenes default attribute tokens need a lookup table; simplified here)
        moving = any(attr in ['moving', 'moving.forward', 'moving.fast'] for attr in attr_tokens)
        standing = any(attr in ['standing', 'not.moving', 'stationary'] for attr in attr_tokens)
        parked = any(attr in ['parked', 'not.moving', 'stationary'] for attr in attr_tokens)
        
        # Tally
        object_summary[category]['count'] += 1
        object_summary[category]['bbox'].append(bbox)
        if moving:
            object_summary[category]['moving'] += 1
        elif standing:
            object_summary[category]['standing'] += 1
        elif parked:
            object_summary[category]['parked'] += 1
    
    return object_summary

def build_llava_conversation(image_path, summary_dict):
    """
    Build a LLaVA conversation with three prompts for a single image and its object summary.
    """

    # Convert summary_dict to readable string
    
    count_map = defaultdict(int)
    for category in summary_dict:
        count_map[category] = summary_dict[category]['count']
        
    bbox_dict = defaultdict(dict)
    for category in summary_dict:
        bbox_dict[category] = {
            'count':summary_dict[category]['count'],
            'bbox':summary_dict[category]['bbox']
        }
    
    count_prompt = "Describe the number of each object present in the scene using the following summary:\n" + str(count_map)    
    count_caption = "This scene has " + str(count_map)
    
    position_prompt = (
        "Based on the object bounding boxes in this summary, describe their approximate relative positions "
        "(e.g., 'a pedestrian to the left of a car', 'a bicycle behind a vehicle'). Summary:\n" + str(bbox_dict)
    )

    attribute_prompt = (
        "Using the summary below, describe the attributes of the objects in the scene such as their actions or states "
        "(e.g., 'a parked car', 'a standing pedestrian'). Summary:\n" + str(summary_dict)
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": count_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": position_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": attribute_prompt},
            ],
        },
    ]

    return conversation, count_caption


def generate_captions(model, processor, dataset:NuScenesObjectDetectDataset, output_path:str):
    
    global_captions = defaultdict(lambda:defaultdict(list))
    
    for idx, data in tqdm(enumerate(dataset)):
        
        summary_dict = summarize_scene_objects(data['labels_2d_cam_front'])        
        conversation, count_caption = build_llava_conversation(data['cam_front_fp'], summary_dict)
        
        global_captions[data['sample_token']]['simple_captions'].append(count_caption)
        
        for message in conversation:

            inputs = processor.apply_chat_template(
                [message],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device, torch.float16)            

            generate_ids = model.generate(**inputs, max_new_tokens=150)
            generated_caption = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            
            generated_caption = generated_caption.split('ASSISTANT:')[1]
            global_captions[data['sample_token']]['llava_generated_captions'].append(generated_caption)

        if (idx + 1) % 10 == 0:        
            with open(f'{output_path}/global_captions.json', 'w+') as f:
                json.dump(global_captions, f)
        

if __name__ == "__main__":

    table_blob_paths = ['../trainval03_blobs_US/tables.json']
    output_path = '../trainval03_blobs_US'

    dataset = create_dataloader(table_blob_paths)
    
    model, processor = load_llava_hf()

    generate_captions(
        model=model, processor=processor, 
        dataset=dataset, 
        output_path=output_path
    )

# inputs = processor.apply_chat_template(
#     conversation,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device, torch.float16)

# # Generate
# generate_ids = model.generate(**inputs, max_new_tokens=50)
# decoded_output = processor.batch_decode(generate_ids, skip_special_tokens=True)

# print(decoded_output)
