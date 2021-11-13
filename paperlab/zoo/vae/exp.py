import torch
from torch.utils.data import DataLoader
from .models import VAEModel
from .data import get_data, collate_fn
from paperlab.core import Config, evaluate_loss, wrap_data


sample_params = {
    'model.dim_latent': 2,
    'model.dim_input': 28 * 28,
    'model.dim_hidden': 500,
    'learning.batch_size': 100,
    'learning.lr': 0.01,
    'learning.num_epoch': 3,
}

sample_config = Config(**sample_params)
MOVING_DECAY = 0.9


def exp(config, return_model=False):
    model = VAEModel(config.model.dim_latent,
                     config.model.dim_input,
                     config.model.dim_hidden)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adagrad(params=model.parameters(), lr=config.learning.lr)
    train_dataset, test_dataset = get_data()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.learning.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.learning.batch_size,
                                 collate_fn=collate_fn)

    train_elbos, test_elbos = {}, {}
    step = 0
    moving_avg_loss = 0
    for _ in range(config.learning.num_epoch):
        for data in train_dataloader:
            if torch.cuda.is_available():
                data = wrap_data(data)

            step += 1
            optimizer.zero_grad()
            loss = model.compute_loss(data)
            loss.backward()
            optimizer.step()

            if step == 1:
                moving_avg_loss = loss.detach().data.item()
            else:
                moving_avg_loss = (1 - MOVING_DECAY) * loss.detach().data.item() + MOVING_DECAY * moving_avg_loss

            train_elbos[step * config.learning.batch_size] = - moving_avg_loss

        # evaluate model after each epoch
        test_loss = evaluate_loss(model, test_dataloader)
        test_elbos[step * config.learning.batch_size] = - test_loss

    if return_model:
        return model, train_elbos, test_elbos
    else:
        return train_elbos, test_elbos
