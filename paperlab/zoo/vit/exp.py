import torch
import einops
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple

from paperlab.core import Config, evaluate_loss, wrap_data
from .models import MultiHeadAttention, ViTClassifier
from .data import get_data


sample_params = {
    'image_size': (32, 32),
    'patch_size': (8, 8),
    'num_channel': 3,
    'pool': 'cls',
    'num_class': 10,
    'use_dataset': 'tiny-imagenet-200',

    'transformer.depth': 4,
    'transformer.dim': 64,
    'transformer.dropout': 0.,
    'transformer.emb_dropout': 0.,
    'transformer.num_head': 4,
    'transformer.dim_head': 32,
    'transformer.dim_mlp': 32,

    'learning.batch_size': 16,
    'learning.lr': 1e-3,
    'learning.num_epoch': 4,
    'learning.early_stop_patience': 5,
    
    'display_freq': 2500,
    'valdate_freq': 10000
}

sample_config = Config(**sample_params)
MOVING_DECAY = 0.9
EPS = 1e-5

def evaluate_accuracy(model, dataloader):
    labels, preds = [], []
    with torch.no_grad():
        for data in dataloader:
            if torch.cuda.is_available():
                data = wrap_data(data)

            image, label = data
            preds.append(model.pred(image))
            labels.append(label)

    acc_score = torch.sum(torch.cat(preds, dim=0) == torch.cat(labels, dim=0)) / len(dataloader.dataset)
    return acc_score.item()


def train(config):
    """
    train a Vision Transformer Image Classifier
    :param config:
    :return: the trained ViT Model, training log
    """
    model = ViTClassifier(
        num_class=config.num_class,
        pool=config.pool,
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_channel=config.num_channel,
        depth=config.transformer.depth,
        dim=config.transformer.dim,
        dropout=config.transformer.dropout,
        emb_dropout=config.transformer.emb_dropout,
        num_head=config.transformer.num_head,
        dim_head=config.transformer.dim_head,
        dim_mlp=config.transformer.dim_mlp,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"number of model parameter: {num_params}")

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=config.learning.num_epoch / 5,
                                                gamma=0.5,
                                                verbose=True,
                                                )

    train_dataset, dev_dataset = get_data(config.use_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.learning.batch_size,
                                  num_workers=4,
                                  shuffle=True)

    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.learning.batch_size,
                                num_workers=4)

    step = 0
    moving_avg_loss = 0
    best_dev_loss, best_dev_score, best_model_state = float('inf'), float('-inf'), deepcopy(model.state_dict())
    patience_cnt = 0
    stats = defaultdict(dict)

    for _ in range(config.learning.num_epoch):
        for data in train_dataloader:
            data = wrap_data(data) if torch.cuda.is_available() else data

            step += 1
            optimizer.zero_grad()
            loss = model.compute_loss(data)
            loss.backward()
            optimizer.step()

            if step == 1:
                moving_avg_loss = loss.detach().data.item()
            else:
                moving_avg_loss = (1 - MOVING_DECAY) * loss.detach().data.item() + MOVING_DECAY * moving_avg_loss

            if step % config.display_freq == 0:
                print(f"step-{step}: training_loss: {moving_avg_loss:.4f}")
                stats['training_loss'][step] = moving_avg_loss
            
            if step % config.validate_freq == 0:
                dev_loss = evaluate_loss(model, dev_dataloader)
                dev_score = evaluate_accuracy(model, dev_dataloader)

                stats['dev_loss'][step], stats['dev_acc'][step] = dev_loss, dev_score

                print(f"step-{step}: dev_loss: {dev_loss:.4f}, dev_acc: {dev_score:.4f}")
                if dev_score > best_dev_score + EPS:
                    best_model_state = deepcopy(model.state_dict())
                    best_dev_score = dev_score

                if dev_loss < best_dev_loss - EPS:
                    patience_cnt = 0
                    best_dev_loss = dev_loss
                else:
                    patience_cnt += 1

                if patience_cnt >= config.learning.early_stop_patience:
                    print(f"dev_loss doesnt descent in {config.learning.early_stop_patience} times validation, halt the training process.")
                    model.load_state_dict(best_model_state)
                    return model, stats
        
        scheduler.step()
    model.load_state_dict(best_model_state)
    return model, stats


def get_attention_distance(model: ViTClassifier, dataloader: DataLoader) -> torch.Tensor:
    """
    compute the mean attention distance as described in Section 4.5 `INSPECTING VISION TRANSFORMER` and  Appendix D.7
    :param model:
    :param dataloader:
    :return: mean attention distance
                shape: [num_layer, num_head]
    """
    if torch.cuda.is_available():
        model = model.cuda()

    # enable cache so that we can store the attention maps across all layers for each image
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.enable_cache()

    # run transformer
    with torch.no_grad():
        for image, _ in dataloader:
            if torch.cuda.is_available():
                image = wrap_data(image)

            model.transformer_encoder(image)

    # retrieve the cached attention map
    attn_maps: List[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            # attn_maps elem size: [data_size, num_head, num_patch, num_patch]
            attn_maps.append(torch.cat(module.cache['attn_map'], dim=0))
            module.disable_cache()
            module.clear_cache()

    _, _, height, width = image.shape
    ph, pw = model.transformer_encoder.patch_height, model.transformer_encoder.patch_width
    nh, nw = height // ph, width // pw

    num_layer, num_head = len(attn_maps), attn_maps[0].shape[1]

    def _pixel_dist_matrix(h, w):
        # the euclid distance between two pixels (h1, w1) and (h2, w2)
        dist = torch.empty(size=(h, w, h, w))
        xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w))
        for x in range(h):
            for y in range(w):
                # the distance between (x, y) and all other pixels, shape: [h, w]
                d = torch.sqrt((x - xx) ** 2 + (y - yy) ** 2)
                dist[x, y] = d

        return dist
    pixel_distance = _pixel_dist_matrix(height, width).to(attn_maps[0].device)

    mean_attention_distance = torch.empty((num_layer, num_head), device=attn_maps[0].device)
    for i in range(num_layer):
        for j in range(num_head):
            # normalize attention value after removing [cls] token
            normalized_attn = attn_maps[i][:, j, 1:, 1:] / torch.sum(attn_maps[i][:, j, 1:, 1:], dim=-1, keepdim=True)
            # the attention pixel (h1, w1) attended to (h2, w2), shape: [data_size, height, width, height, width]
            head_attn_pixel = einops.repeat(normalized_attn / (ph * pw),
                                            'b (nhx nwx) (nhy nwy) -> b (nhx phx) (nwx pwx) (nhy phy) (nwy pwy)',
                                             nhx=nh, nwx=nw, nhy=nh, nwy=nw,
                                             phx=ph, pwx=pw, phy=ph, pwy=pw)
            # shape: [data_size, height, width, height, width]
            weighted_distance = head_attn_pixel * torch.unsqueeze(pixel_distance, dim=0)
            mean_attention_distance[i, j] = torch.mean(torch.sum(weighted_distance, dim=(-1, -2)))

    return mean_attention_distance


def get_attention_maps(model: ViTClassifier, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    get the attention maps (i.e. the attention on other token queried by the [CLS] token)
        over the images in the dataloader
        described in Section 4.5 `INSPECTING VISION TRANSFORMER` and Appendix D.8
    :param model:
    :param dataloader:
    :return: tuple (attn_map_pixel, images)
                shape: [data_size, height, width], [data_size, num_channel, height, width]
    """
    assert model.pool == 'cls', 'only model with `cls` pooling method can generate attention map'

    if torch.cuda.is_available():
        model = model.cuda()

    # enable cache so that we can store the attention maps across all layers for each image
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            module.enable_cache()

    # let transformer process images
    images = []
    with torch.no_grad():
        for image, _ in dataloader:
            if torch.cuda.is_available():
                image = wrap_data(image)

            model.transformer_encoder(image)
            images.append(image)

    images = torch.cat(images, dim=0)  # [data_size, num_channel, height, width]

    # retrieve the cached attention map
    attn_maps = []
    for module in model.modules():
        if isinstance(module, MultiHeadAttention):
            # average attention over each head, shape: [data_size, num_patch, num_patch]
            avg_attn_map = torch.mean(torch.cat(module.cache['attn_map'], dim=0), dim=1)
            attn_maps.append(avg_attn_map)
            module.disable_cache()
            module.clear_cache()

    rollout = attn_rollout(attn_maps)  # [data_size, num_patch, num_patch]
    # get attention for [cls] to each input patch token
    # [data_size, num_patch - 1]
    attn_query_by_cls = rollout[:, 0, 1:] / torch.sum(rollout[:, 0, 1:], dim=1, keepdim=True)

    _, _, height, width = images.shape
    ph, pw = model.transformer_encoder.patch_height, model.transformer_encoder.patch_width
    nh, nw = height // ph, width // pw
    # repeat patch tensor to pixel
    # [data_size, height, width]
    attn_map_pixel = einops.repeat(attn_query_by_cls,
                                   'b (nh nw) -> b (nh ph) (nw pw)',
                                   nh=nh, nw=nw,
                                   ph=ph, pw=pw)

    return attn_map_pixel, images


def attn_rollout(attn_matrices: List[torch.Tensor]):
    """
    get the attention flow of each output unit at the last layer to each input token at the first layer in transformer 
    based on the attention rollout algorithm described in `Quantifying Attention Flow in Transformers`
    :param attn_matrices: the attention matrix at each layer in transformer
            element shape: [batch_size, num_token, num_token]
    :return: shape: [batch_size, num_token, num_token]
    """
    b, n, _ = attn_matrices[0].shape
    device = attn_matrices[0].device
    rollout = einops.repeat(torch.eye(n, device=device), 'n m -> b n m', b=b)
    for attn in map(lambda x: 0.5 * x + 0.5 * torch.eye(n, device=device), attn_matrices):
        rollout = torch.einsum('bij, bjk -> bik', rollout, attn)
    return rollout