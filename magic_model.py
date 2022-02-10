import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 patch_embs,
                 idx_ls,
                 num_tokens,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 mlp_ratio= 4,
                 ) -> None:
        super().__init__()
        
        self.patch_embs =patch_embs
        self.idx_ls =idx_ls
        self.num_tokens = num_tokens
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        self.pos_embedding = torch.nn.Parameter(torch.randn((self.num_tokens, 1, emb_dim)))
#         self.pos_embedding = torch.nn.Parameter(torch.zeros(1,(image_size // patch_size) ** 2, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

#         self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        
#         self.patch_embs = torch.nn.Linear(patch_size**2*3, emb_dim)
        
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,mlp_ratio=mlp_ratio) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_channel = x.size(1)
        num_patches = len(self.patch_embs)
        x = x.view(batch_size, num_channel, -1)
        # x (b, c, h * w)
        mlp_out = []
#         print(x.shape)
        for ni, l in enumerate(self.patch_embs):
#             idx = torch.nonzero(self.labels==self.valid_labels[ni]).squeeze().to(x.device)
            idx = self.idx_ls[ni].to(x.device)
            ix = torch.index_select(x, 2, idx).permute(0,2,1).reshape(batch_size, -1).to(x.device)
            ih = l(ix).unsqueeze(0)
            mlp_out.append(ih)
            
        patches = torch.cat(mlp_out, dim=0)
#         print(out.shape)
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)
#         print(patches.shape)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features,backward_indexes
    
    def get_feature(self, x):
        batch_size = x.size(0)
        num_channel = x.size(1)
        num_patches = len(self.patch_embs)
        x = x.view(batch_size, num_channel, -1)
        # x (b, c, h * w)
        mlp_out = []
    #         print(x.shape)
        for ni, l in enumerate(self.patch_embs):
            idx = self.idx_ls[ni].to(x.device)
            ix = torch.index_select(x, 2, idx).permute(0,2,1).reshape(batch_size, -1).to(x.device)
            ih = l(ix).unsqueeze(0)
            mlp_out.append(ih)

        patches = torch.cat(mlp_out, dim=0)
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        return features.mean(1)
    
class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 patch_embs_r,
                 idx_ls,
                 num_tokens,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 mlp_ratio = 4,
                 ) -> None:
        super().__init__()
        
        self.patch_embs_r =patch_embs_r
        self.idx_ls =idx_ls
        self.num_tokens = num_tokens
        self.image_size =image_size
        
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.randn((num_tokens + 1, 1, emb_dim)))
#         self.pos_embedding = torch.nn.Parameter(torch.zeros(1,(image_size // patch_size) ** 2, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,mlp_ratio=mlp_ratio) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        B = features.shape[1]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
#         print(features.shape,self.pos_embedding.shape)
#         features = features + self.pos_embedding
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature
#         print(features.shape)
        patches = self.head(features)
#         print(features.shape,patches.shape)
        template = torch.zeros((B,3,self.image_size**2)).to(features.device)
        template_mask = torch.zeros((B,3,self.image_size**2)).type(torch.long).to(features.device)
#         print(template.shape)
#         print(template.shape)
        mask = torch.zeros_like(backward_indexes[1:]).to(features.device)
#         print(mask.shape,T)
        mask[T:] = 1
        mask = torch.gather(mask, 0, backward_indexes[1:] - 1)
#         print(mask.shape)
        
        for ni, l in enumerate(self.patch_embs_r):
#             idx = torch.nonzero(self.labels==self.valid_labels[ni]).squeeze().to(x.device)
            idx = self.idx_ls[ni].to(features.device)
#             print(idx)
#             print(torch.index_select(x, 2, idx).shape)
#             break
#             print(idx.shape)
#             print(template[:,:,idx].shape,l(features[ni]).reshape(B,3,-1).shape)

            each_patch = l(features[ni])
#             print(each_patch.shape)
            each_patch_expand = each_patch.reshape(B,3,-1)
            template[:,:,idx] = each_patch_expand
#             print(template_mask[:,:,idx].shape,mask[ni].shape)
            
            template_mask[:,:,idx] = repeat(mask[ni], 'b -> b c t', c=3, t = each_patch_expand.shape[-1])
        
        
            
#         img = self.patch2img(patches)
#         mask = self.patch2img(mask)

        return template.unfold(-1, self.image_size, self.image_size), template_mask.unfold(-1, self.image_size, self.image_size)
    
class MAE_ViT_magic(torch.nn.Module):
    def __init__(self,
                 labels,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 mlp_ratio = 4,
                 ) -> None:
        super().__init__()
        
        self.patch_size = patch_size
        self.image_size = image_size
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.patch = image_size // patch_size

        # labels: 32 x 32 and flatten
        labels_np = np.load(labels)
        self.labels = torch.from_numpy(labels_np).type(torch.long).view(-1)
        self.valid_labels = []
        MLPlist = []
        MLPlist_r = []
        idx_ls = []
        for i in range(self.patch**2):
            idx = torch.nonzero(self.labels==i).squeeze()
            if idx.size(0)!= 0:
                # MLPlist.append(nn.Linear(3*idx.size(0), hidden))
                MLPlist.append(torch.nn.Linear(3*idx.size(0), emb_dim))
                MLPlist_r.append(torch.nn.Linear(emb_dim,3*idx.size(0)))
#                 idx = torch.nonzero(self.labels==i).squeeze()
                idx_ls.append(idx)
                self.valid_labels.append(i)
            else:
                print("label", i, "is an empty cluster") 
                
        patch_embs = torch.nn.ModuleList(MLPlist)
        patch_embs_r = torch.nn.ModuleList(MLPlist_r)
        
        print("num of nonempty clusters:", len(self.valid_labels), "num of mlps:", len(patch_embs))
        num_tokens =  len(self.valid_labels)

        self.encoder = MAE_Encoder(patch_embs, idx_ls, num_tokens, image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio, mlp_ratio)
        self.decoder = MAE_Decoder(patch_embs_r, idx_ls, num_tokens, image_size, patch_size, emb_dim, decoder_layer, decoder_head, mlp_ratio)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask
    
            

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
#         self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits



if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    print(loss)