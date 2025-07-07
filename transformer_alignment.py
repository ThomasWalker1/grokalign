import torch
import numpy as np
from tqdm import tqdm
from utils import Transformer, GrokAlign, Centroids, full_loss, full_accuracy, gini_from_fourier_norms
import wandb as wb
import os
import argparse

def train(config, device='cuda'):

    run_name = f"{config.p}-{config.weight_decay}Wd-{config.lambda_jac}GA-{'FixEmb' if config.fixed_embedding else ''}"
    run = wb.init(project='test_transformer_alignment', config=vars(config), name=run_name)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    output_directory = './outputs/transformer_alignemnt'
    run_directory = f'{output_directory}/{run_name}'
    os.makedirs(run_directory, exist_ok=True)

    equals_token = config.p
    x, y = torch.meshgrid(torch.arange(config.p), torch.arange(config.p), indexing='ij')
    x = x.flatten()
    y = y.flatten()

    equals = torch.ones(x.shape, dtype=torch.int64) * equals_token
    prompts = torch.stack([x, y, equals], dim=1).to(device)
    answers = ((x + y) % config.p).to(device)

    data = torch.utils.data.TensorDataset(prompts, answers)
    train, test = torch.utils.data.random_split(data, [
        int(config.fraction * len(data)),
        len(data) - int(config.fraction * len(data))
    ])

    train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), shuffle=False)

    loaders = {'train': train_loader, 'test': test_loader}

    model = Transformer(
        d_vocab=equals_token + 1,
        d_model=config.d_model,
        d_mlp=config.d_mlp,
        d_head=config.d_head,
        num_heads=config.num_heads,
        n_ctx=config.n_ctx
    ).to(device)

    if config.lambda_jac > 0:
        grokalign = GrokAlign(lambda x: model.unembed(model.block(model.pos_embed(x)))[:, -1],device=device)
    embedding_centroids = Centroids(lambda x: model.unembed(model.block(model.pos_embed(x)))[:, -1], device=device)

    if config.fixed_embedding:
        excluded_params = {id(model.embed.W_E)}
    else:
        excluded_params = {}
    params_to_optimize = [p for p in model.parameters() if id(p) not in excluded_params]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=config.lr, weight_decay=config.weight_decay, betas=config.betas)

    logged_steps = np.unique(np.append(np.logspace(0, np.log10(config.steps), config.num_logs, dtype=int), [0, config.steps]))
    logged_steps = np.sort(logged_steps)
    print(logged_steps)

    pbar = tqdm(range(config.steps + 1))
    for step in pbar:
        if step in logged_steps:
            model.eval()
            stats = {'step': step}
            for set_type, loader in loaders.items():
                with torch.no_grad():
                    loss = full_loss(model, loader, device).item()
                    accuracy = full_accuracy(model, loader, device)
                    stats[f'{set_type}_accuracy'] = accuracy
            x=model.embed(next(iter(train_loader))[0])
            alignments = embedding_centroids(x).get_alignments()
            stats['centroid_alignment_min'] = alignments.min().item()
            stats['centroid_alignment_mean'] = alignments.mean().item()
            stats['centroid_alignment_max'] = alignments.max().item()
            with torch.no_grad():
                stats['gini_WE'] = gini_from_fourier_norms(model.embed.W_E)
                stats['gini_WL'] = gini_from_fourier_norms(((model.unembed.W_U.T).to(device) @ model.block.mlp.W_out).T)

            wb.log(stats)

        model.train()
        train_loss = full_loss(model, train_loader, device)
        if config.lambda_jac > 0:
            x=model.embed(next(iter(train_loader))[0])
            train_loss += config.lambda_jac * grokalign(x)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f'{step} - {train_loss.item():.4f}')

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=113)
    parser.add_argument('--fraction', type=float, default=0.3)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_mlp', type=int, default=512)
    parser.add_argument('--d_head', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--n_ctx', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1.0)
    parser.add_argument('--fixed_embedding', action='store_true')
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--steps', type=int, default=48000)
    parser.add_argument('--num_logs', type=int, default=48)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--betas', type=lambda s: tuple(map(float, s.strip("()").split(","))), default=(0.9, 0.98))

    args = parser.parse_args()
    train(args, device=args.device)
