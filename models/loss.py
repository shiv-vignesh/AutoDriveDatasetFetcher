
import torch

class CLIPLoss(torch.nn.Module):
    
    def forward(self, text_features:torch.Tensor, #(bs, N_t, dim)
                image_features:torch.Tensor, #(bs, N_i, dim)
                logit_scale):

        if len(text_features.shape) == 3:        
            bs, N_t, c = text_features.shape
            text_features = text_features.reshape(bs * N_t, c)

        if len(image_features.shape) == 3:
            bs, N_i, c = image_features.shape
            image_features = image_features.reshape(bs * N_i, c)

        logits_per_text = torch.matmul(text_features, image_features.t().to(text_features.device))
        logits_per_text = logits_per_text * logit_scale.to(text_features.device)

        logits_per_image = logits_per_text.t()

        labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device, dtype=torch.long)

        caption_loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        image_loss = torch.nn.functional.cross_entropy(logits_per_text, labels)

        total_loss = (caption_loss + image_loss) / 2.0

        return total_loss

    # def forward(self, text_features:torch.Tensor, #(bs, N_t, dim)
    #             image_features:torch.Tensor, #(bs, N_i, dim)
    #             similarity_matrix:torch.Tensor,
    #             logit_scale):
        
    #     logits_per_image = logit_scale * image_features @ text_features.T  # [B, B]
    #     logits_per_text = logits_per_image.T  # [B, B]

    #     labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)

    #     # Invert similarity matrix to get "difficulty" of negatives
    #     neg_weights = 1.0 - similarity_matrix.clamp(0, 0.99)

    #     # Downweight soft negatives in logits
    #     mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        
    #     weighted_logits_i2t = logits_per_image - (1 - mask) * (1 - neg_weights) * 10.0
    #     weighted_logits_t2i = logits_per_text - (1 - mask) * (1 - neg_weights.T) * 10.0

    #     # Contrastive loss
    #     loss_i2t = torch.nn.functional.cross_entropy(weighted_logits_i2t, labels)
    #     loss_t2i = torch.nn.functional.cross_entropy(weighted_logits_t2i, labels)
    #     loss = (loss_i2t + loss_t2i) / 2        
        
    #     return loss