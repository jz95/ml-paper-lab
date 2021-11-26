from unittest import TestCase
from paperlab.zoo.vit import *
from torch.utils.data import DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



class TestViT(TestCase):
    def setUp(self) -> None:
        from paperlab.zoo.vit.exp import sample_config
        config = sample_config
        self.config = config
        self.model = ViTClassifier(
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



    def test_data(self):
        from paperlab.zoo.vit.data import get_data
        trainset, *_ = get_data('tiny-imagenet-200')
        image, label = trainset[-1]
        
        print(f"train-size: {len(trainset)}, image-shape:{image.shape}, label:{label}")

    def test_attn(self):
        _, test_dataset = get_data()
        from torch.utils.data import Subset
        dataset = Subset(test_dataset, indices=[0, 1, 2, 3, 4, 5])
        sample_dataloader = DataLoader(dataset,
                                       batch_size=self.config.learning.batch_size)

        attn_map_pixel, images = get_attention_maps(self.model, sample_dataloader)
        self.assertListEqual(list(attn_map_pixel.size()), list(images.transpose(1, 2).transpose(2, 3).size())[: -1])

    def test_attn_dist(self):
        _, test_dataset = get_data()
        from torch.utils.data import Subset
        dataset = Subset(test_dataset, indices=[0, 1, 2, 3, 4, 5])
        sample_dataloader = DataLoader(dataset,
                                       batch_size=self.config.learning.batch_size)

        get_attention_distance(self.model, sample_dataloader)


if __name__ == '__main__':
    import unittest

    unittest.main()