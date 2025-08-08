import torch
from transformers import AutoModel, CLIPModel

from .loss import CLIPLoss

class CLIP(torch.nn.Module):
    def __init__(self, text_model_name:str="bert_model", vision_model_name:str="vit_model",
                text_model_device=torch.device("cpu"), vision_model_device=torch.device("cpu"),
                projection_dim:int=512):
        super(CLIP, self).__init__()
        
        self.text_model_name = text_model_name
        self.vision_model_name = vision_model_name
        
        if text_model_name == "bert_model":
            self.text_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        if vision_model_name == "vit_model":
            self.vision_model = AutoModel.from_pretrained("google/vit-base-patch16-224")
            
        self.text_model_device = text_model_device
        self.vision_model_device = vision_model_device
        
        self.projection_dim = projection_dim

        self.visual_projection = torch.nn.Linear(self.vision_model.config.hidden_size, self.projection_dim, bias=False)
        self.text_projection = torch.nn.Linear(self.text_model.config.hidden_size, self.projection_dim, bias=False)
        
        self.text_model.to(text_model_device)
        self.vision_model.to(vision_model_device)
        
        self.visual_projection.to(vision_model_device)
        self.text_projection.to(text_model_device)
        
        self._init_log_scale = torch.log(torch.tensor(1/0.07))

        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0) * self._init_log_scale)
        
    def forward(self, pixel_values:torch.Tensor, input_ids:torch.Tensor,
                attention_masks:torch.Tensor, return_loss:bool=True):
        
        text_outputs = self.text_model(input_ids=input_ids.to(self.text_model_device),
                                  attention_mask=attention_masks.to(self.text_model_device))
        if hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output  # [B, D]
            text_features = self.text_projection(text_features)
        else:
            text_features = text_outputs.last_hidden_state        
        
        vision_outputs = self.vision_model(pixel_values=pixel_values.to(self.vision_model_device))
        
        if hasattr(vision_outputs, "pooler_output"):
            image_features = vision_outputs.pooler_output  # [B, D]
            image_features = self.visual_projection(image_features)
        else:
            image_features = vision_outputs.last_hidden_state

        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()

        if return_loss:
            loss = CLIPLoss()(
                text_features.to(self.vision_model_device),
                image_features,
                logit_scale.to(self.vision_model_device)
            )
        else:
            loss = None

        return {
            "text_features":text_features,
            "image_features":image_features,
            "loss":loss
        }