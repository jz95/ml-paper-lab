"""
Code Base: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""
import torch
from torch import nn
from paperlab.core import BaseModel
from typing import Union, Tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def add_cache(cls):
    class ModuleWrapper(cls):
        def __init__(self, *args, **kwargs):
            super(ModuleWrapper, self).__init__(*args, **kwargs)
            self.__class__.__name__ = f"{cls.__name__}WithCache"
            self.cache = {}
            self.cache_enable = False

        def enable_cache(self):
            self.cache_enable = True

        def disable_cache(self):
            self.cache_enable = False

        def clear_cache(self):
            self.cache.clear()

    return ModuleWrapper


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hidden, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


@add_cache
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_head=8, dim_head=64, dropout=0.):
        super().__init__()
        assert dim % num_head == 0, 'Multi Head Attention dimensionality must be divisible by the number of heads.'
        project_out = not (num_head == 1 and dim_head == dim)

        self.num_head = num_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, num_head * dim_head * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(num_head * dim_head, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    def forward(self, x):
        """
        :param x: shape: [batch_size, num_patch, dim]
        :return: encoded output, shape: [batch_size, num_patch, dim]
        """
        # do the attention process within each head in parallel
        # [batch, num_patch, num_head * dim_head, 3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # q, k, v: [batch_size, num_head, num_patch, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_head), qkv)

        # [batch_size, num_head, num_patch, num_patch]
        attn = self.attend(torch.einsum('bhnd, bhmd -> bhnm', q, k) * self.scale)
        # [batch_size, num_head, num_patch, dim_head]
        head_out = torch.einsum('bhnm, bhmd -> bhnd', attn, v)

        # concat head-out in the head-dimensionality, parallel process done
        out = rearrange(head_out, 'b h n d -> b n (h d)')

        if self.cache_enable:
            if 'attn_map' not in self.cache:
                self.cache['attn_map'] = []

            self.cache['attn_map'].append(attn)

        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_head, dim_head, dim_mlp, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention(dim, num_head, dim_head, dropout)
        self.ffn = FeedForward(dim, dim_mlp, dropout)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.layer_norm_ffn = nn.LayerNorm(dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_patch, dim]
        :return: [batch_size, num_patch, dim]
        """
        x = self.multi_head_attn(self.layer_norm_attn(x)) + x
        return self.layer_norm_ffn(self.ffn(x)) + x


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: Union[Tuple[int, int], int],
                 patch_size: Union[Tuple[int, int], int],
                 num_channel,
                 depth,
                 dim,
                 num_head,
                 dim_head,
                 dim_mlp,
                 dropout=0.,
                 emb_dropout=0.):
        """
        :param image_size: image resolution, both tuple and int are acceptable, tuple structure (height, width),
                           if given a int parameter, the resolution will be interpreted as (size, size)
        :param patch_size: patch resolution, both tuple and int are acceptable
        :param num_channel: channel number of input image
        :param depth: number of transformer blocks
        :param dim: the dimensionality for each transformer block
        :param num_head: number of head in the multi-head attention component
        :param dim_head:
        :param dim_mlp: dimensionality of hidden layer within the ffn in transformer block
        :param dropout: dropout rate on all linear layers in the transformer
        :param emb_dropout: dropout rate on patch embedding
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patch = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = num_channel * patch_height * patch_width

        self.patch_height, self.patch_width = patch_height, patch_width

        # transform a batch of images to a sequence of patch tokens
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)', ph=patch_height, pw=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_patch + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, num_head, dim_head, dim_mlp, dropout) for _ in range(depth)]
        )

    def forward(self, img):
        """
        :param img: input image, shape: [batch_size, num_channel, height, width]
        :return:
        """
        # [batch_size, num_patch, dim]
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # [batch_size, 1, dim]
        cls_tokens = torch.unsqueeze(repeat(self.cls_token, 'd -> b d', b=b), dim=1)
        # [batch_size, n + 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)
        x += torch.unsqueeze(self.pos_embedding[: (n+1)], dim=0)
        x = self.dropout(x)

        return self.transformer(x)

class ViTClassifier(BaseModel):
    """
    Vision Transformer based image classifier
    """
    def __init__(self,
                 num_class,
                 pool: str,
                 image_size: Union[Tuple[int, int], int],
                 patch_size: Union[Tuple[int, int], int],
                 num_channel,
                 depth,
                 dim,
                 num_head,
                 dim_head,
                 dim_mlp,
                 dropout=0.,
                 emb_dropout=0.):
        """

        :param num_class:
        :param pool: pooling method for the transformer output either 'cls' or 'mean'
        :param image_size: image resolution, both tuple and int are acceptable, tuple structure (height, width),
                           if given a int parameter, the resolution will be interpreted as (size, size)
        :param patch_size: patch resolution, both tuple and int are acceptable
        :param num_channel: channel number of input image
        :param depth: number of transformer blocks
        :param dim: the dimensionality for each transformer block
        :param num_head: number of head in the multi-head attention component
        :param dim_head:
        :param dim_mlp: dimensionality of hidden layer within the ffn in transformer block
        :param dropout: dropout rate on all linear layers in the transformer
        :param emb_dropout: dropout rate on patch embedding
        """
        super(ViTClassifier, self).__init__()
        self.transformer_encoder = VisionTransformer(image_size, patch_size, num_channel, depth, dim, num_head, dim_head, dim_mlp, dropout, emb_dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.pred_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_class)
        )
        assert pool in ('cls', 'mean'), 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

    def _get_pred_layer_out(self, x):
        """
        :param x: [batch_size, height, width, channel]
        :return: [batch_size, num_class]
        """
        out = self.transformer_encoder(x)
        out = out[:, 0, :] if self.pool == 'cls' else torch.mean(out, dim=1)
        return self.pred_layer(out)

    def compute_loss(self, data, reduction='mean') -> torch.Tensor:
        image, label = data
        self.criterion.reduction = reduction
        return self.criterion(self._get_pred_layer_out(image), label)

    def pred_prob(self, x):
        return torch.functional.F.softmax(self._get_pred_layer_out(x), dim=-1)

    def pred(self, x):
        return torch.argmax(self._get_pred_layer_out(x), dim=-1)