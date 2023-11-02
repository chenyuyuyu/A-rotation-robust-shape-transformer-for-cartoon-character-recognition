import torch.nn as nn
import torch
from einops import rearrange, repeat
import numpy as np
import torch.nn.functional as F
from py.transformer import Transformer


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self,x):
        return x+self.pos_table[:, :x.size(1)].clone().detach()

class scTransformer(nn.Module):
    def __init__(self,num_classes,heads,depth,mlp_dim,dist=False,pool = 'cls',dim_head = 64,dim=732,npoints=150,dropout = 0.,emb_dropout = 0.):
        super(scTransformer, self).__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # self.encoder_layer=nn.TransformerEncoderLayer(d_model=dim,nhead=heads)
        self.dist=dist
        # self.transformer_encoder=nn.TransformerEncoder(self.encoder_layer,num_layers=layers)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        if self.dist is False:
            # self.pos_embedding = nn.Parameter(torch.randn(1, npoints + 1, dim))#
            self.pos_embedding = PositionalEncoding(dim,n_position=npoints+1)#
        else:
            self.pos_cls = nn.Parameter(torch.randn(1, 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def dist_embedding(self,x, dist_enc):
        b,n,d=x.shape
        pos_enc = np.empty((b, n, d))
        dist_enc=dist_enc.view(b,n,1)##
        dist_enc=repeat(dist_enc, 'b n () -> b n d', d=d).cpu().numpy()

        for i in range(d):
            pos_enc[:, :, i] = dist_enc[:, :,i] * (n+1) / np.power(10000, 2 * (i // 2) / d)
        pos_enc[:, :, 0::2] = np.sin(pos_enc[:, :, 0::2])  # dim 2i
        pos_enc[:, :, 1::2] = np.cos(pos_enc[:, :, 1::2])  # dim 2i+1
        pos_enc = torch.FloatTensor(pos_enc).cuda()

        return x+pos_enc[:, :n].clone().detach()

    def forward(self,input):

        b, n, _ = input["sc"].shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, sc), dim=1)
        if self.dist is False:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, input["sc"]), dim=1)
            # x += self.pos_embedding[:, :(n + 1)]
            x=self.pos_embedding(x)
        else:
            #dist+angle
            # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            # x = torch.cat((cls_tokens, input["sc"]), dim=1)
            # only dist,no angle
            cls_tokens = repeat(self.cls_token+self.pos_cls, '() n d -> b n d', b=b)
            x = self.dist_embedding(input["sc"], input["dist_enc"])
            x = torch.cat((cls_tokens, x), dim=1)


        x = self.dropout(x)

        # x=self.transformer_encoder(x)
        x = self.transformer(x)
        # x = self.transformer(x,input["dist_enc"],input["angle_enc"])
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        pred=self.mlp_head(x)
        # pred=F.softmax(pred,dim=1)
        return pred
