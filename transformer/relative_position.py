import random
import time

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings #[lenq,lenk,num]


class RelativePositionDA(nn.Module):

    def __init__(self, num_units=64, num=151):
        super().__init__()
        self.n = num
        self.num_units = num_units
        # self.cls_embedding1 = nn.Parameter(torch.randn((1,1,num,1)))
        # self.cls_embedding2 = nn.Parameter(torch.randn((1,num - 1,1,1)))

        # self.cls_embedding1 = nn.Parameter(torch.Tensor(1, 1, num, 1))
        # self.cls_embedding2 = nn.Parameter(torch.Tensor(1, num - 1, 1, 1))
        # nn.init.xavier_uniform_(self.cls_embedding1)
        # nn.init.xavier_uniform_(self.cls_embedding2)
        self.cls_embidding=np.array(random.random()).astype(np.float32).reshape((1,1))

    def forward(self, enc):
        # start=time.time();
        b,_=enc.shape
        enc=enc.cpu().numpy()
        cls_embedding=repeat(self.cls_embidding, '() n -> b n ', b=b)
        enc=np.hstack((cls_embedding, enc))
        # distance_mat = enc[:,None, :] - enc[:,:, None]  # (bs,150,150)
        # distance=np.empty((b,self.n-1,self.n-1,self.num_units)) #(bs,150,150,64)
        # for i in range(self.num_units):
        #     distance[:,:,:,i]=distance_mat[:,:,:]*self.n// np.power(10000, 2 * (i // 2) / self.num_units)
        # distance[:,:,:,0::2]=np.sin(distance[:,:,:,0::2]) #dim 2i
        # distance[:,:,:,1::2]=np.cos(distance[:,:,:,1::2]) #dim 2i+1
        distance = np.empty((b, self.n, self.num_units))
        for i in range(self.num_units):
            distance[:, :, i] = enc[:, :] * self.n // np.power(10000, 2 * (i // 2) / self.num_units)
        distance[:, :, 0::2] = np.sin(distance[:, :, 0::2])  # dim 2i
        distance[:, :, 1::2] = np.cos(distance[:, :, 1::2])  # dim 2i+1
        distance = distance[:, None, :, :] - distance[:, :, None, :]  # (bs,150,150)
        distance=torch.FloatTensor(distance).cuda()
        # cls_embedding1=repeat(self.cls_embedding1, '() n d () -> b n d h', b=b,h=self.num_units) #(bs,1,151,64)
        # cls_embedding2=repeat(self.cls_embedding2, '() n d ()-> b n d h', b=b,h=self.num_units) #(bs,150,1,64)
        # embedding=torch.cat((cls_embedding2, distance), dim=2)
        # embedding=torch.cat((cls_embedding1,embedding),dim=1) #(bs,151,151,64)

        # end=time.time()
        # t=end-start
        # print(t)
        # embedding = embedding.view(b,self.n, self.n, 1)
        # embedding = repeat(embedding, 'b n m () -> b n m d', d=self.num_units) #(bs,151,151,64)
        return distance#embedding #.detach()


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=64, dropout=0.):
        super().__init__()

        # assert hid_dim % n_heads == 0

        self.hid_dim = head_dim * n_heads
        project_out = not (n_heads == 1 and head_dim == dim)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_relative_position = 2

        # self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        # self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_dist = RelativePositionDA(self.head_dim)
        self.relative_position_angle = RelativePositionDA(self.head_dim)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, self.hid_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.hid_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dist_enc=None, angle_enc=None, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)
        batch_size = x.shape[0]
        # len_k = key.shape[2]
        # len_q = query.shape[2]
        # len_v = value.shape[2]

        attn1 = torch.matmul(query, key.permute(0, 1, 3, 2))  # torch.Size([bs, heads, lenq, lenk])

        r_q2 = query.permute(0, 2, 1, 3).contiguous() #(bs,lenq, heads, self.head_dim)
        r_k2 = self.relative_position_dist(angle_enc) #(bs,lenq,lenk,d)
        attn2 = torch.matmul(r_q2, r_k2.transpose(2, 3)).transpose(1, 2)  #(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) * self.scale
        # attn = (attn1 ) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        weight1 = torch.matmul(attn, value) #(batch_size, self.n_heads, len_q, self.head_dim)
        r_v2 = self.relative_position_angle(angle_enc) #(bs,lenq,lenv,d)
        weight2 = attn.permute(0, 2, 1, 3) #(bs, lenq, heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)  #(bs,lenq,heads,d)
        weight2 = weight2.transpose(1, 2)

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.to_out(x)

        # x = [batch size, query len, dim]

        return x
