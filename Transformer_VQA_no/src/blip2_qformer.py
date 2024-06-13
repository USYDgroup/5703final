import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


class Blip2Qformer(Blip2Base):
    def __init__(
        self,
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        embed_dim=256,
        max_txt_len=32,
        vision_model=''
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        
        #加载eva_vit
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        if freeze_vit:
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
            
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
       
if __name__ == '__main__':
    bq = Blip2Qformer() 