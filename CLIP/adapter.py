
import torch
from torch import nn

import torch.nn.functional as F

# Residual CLIP Adapter
class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y


class CLIPAD(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = features
        self.adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        ada_patch_tokens = []
        
        for i, res in enumerate(self.image_encoder.transformer.resblocks):
            x, _ = res(x, attn_mask=None)
            if (i + 1) in self.features:
                adapt_med, adapt_out = self.adapters[self.features.index(i+1)](x)

                x = 0.9 * x + 0.1 * adapt_out
                ada_patch_tokens.append(adapt_med)
            
        x = x.permute(1, 0, 2)

        ada_patch_tokens = [ada_patch_tokens[t].permute(1, 0, 2) for t in range(len(ada_patch_tokens))]

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        return pooled, ada_patch_tokens


        
