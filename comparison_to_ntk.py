import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import argparse
import wandb as wb

from utils import Centroids

def set_global_seed(seed,dtype=torch.float64):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def prepare_dataloaders(k_classes=2, total_points=1024, download_directory='./data'):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.flatten())
    ])
    train = torchvision.datasets.MNIST(root=download_directory, train=True, transform=transform, download=True)
    test = torchvision.datasets.MNIST(root=download_directory, train=False, transform=transform, download=True)

    def extract_subset(dataset):
        class_counts = {cls: 0 for cls in range(k_classes)}
        selected_indices = []
        quota, extra = divmod(total_points, k_classes)
        for idx, (_, label) in enumerate(dataset):
            if label in class_counts and class_counts[label] < quota + (1 if label < extra else 0):
                selected_indices.append(idx)
                class_counts[label] += 1
            if sum(class_counts.values()) == total_points:
                break
        return torch.utils.data.Subset(dataset, selected_indices)

    train_loader = torch.utils.data.DataLoader(extract_subset(train), batch_size=1024, shuffle=False)
    test_loader = torch.utils.data.DataLoader(extract_subset(test), batch_size=256, shuffle=False)
    return {'train': train_loader, 'test': test_loader}

def build_model(input_dim, config):
    layers = [nn.Linear(input_dim, config.width, bias=False), nn.ReLU()]
    for _ in range(config.depth - 2):
        layers += [nn.Linear(config.width, config.width, bias=False), nn.ReLU()]
    layers += [nn.Linear(config.width, 1, bias=False)]
    model = nn.Sequential(*layers).to(config.device)

    with torch.no_grad():
        for p in model.parameters():
            p.mul_(config.init_scale)
    return model

def evaluate(model, loaders, sample_point, config):
    model.eval()
    stats = {'step': config.step}

    for set_type, loader in loaders.items():
        stats[f'{set_type}_accuracy'] = compute_accuracy(model, loader, config)

    losses = compute_individual_ms(model, loaders['train'], config)
    ntks = compute_ntk_with_sample(model, sample_point, loaders['train'], config)
    stats.update({
        'ntk_mean': np.mean(ntks),
        'rate_of_change': compute_rate_of_change_of_alignment(ntks, losses, config.lr)
    })
    data = centroid_statistics(sample_point, Centroids(model))
    stats.update({
        'centroid_norm_reciprocal': 1/data['norm'],
        'centroid_sim_unnormalized': data['unnormalised'],
        'centroid_sim_normalized': data['normalised']
    })
    return stats

def compute_accuracy(model, loader, config):
    correct = total = 0
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(config.device), labels.float().unsqueeze(1).to(config.device)
            preds = torch.sigmoid(config.output_scale * model(x))
            correct += ((preds > 0.5).float() == labels).sum().item()
            total += labels.size(0)
    return correct / total

def compute_individual_ms(model, loader, config):
    losses = []
    with torch.no_grad():
        for x, labels in loader:
            x, labels = x.to(config.device), labels.float().unsqueeze(1).to(config.device)
            output = config.output_scale * model(x)
            losses.append((labels.flatten() - torch.sigmoid(output).squeeze(1)).cpu().numpy())
    return np.concatenate(losses)

def compute_ntk_with_sample(model, sample_point, loader, config):
    sample_point = sample_point.to(config.device).requires_grad_(True)
    out_sample = config.output_scale * model(sample_point).squeeze()
    grad_sample = torch.autograd.grad(out_sample, list(model.parameters()), retain_graph=True, allow_unused=True)
    grad_sample_flat = torch.cat([g.flatten() for g in grad_sample if g is not None])

    ntk_values = []
    for x, _ in loader:
        x = x.to(config.device).requires_grad_(True)
        out = config.output_scale * model(x).squeeze()
        for i in range(len(x)):
            grad_i = torch.autograd.grad(out[i], list(model.parameters()), retain_graph=True, allow_unused=True)
            grad_i_flat = torch.cat([g.flatten() for g in grad_i if g is not None])
            ntk_values.append(torch.dot(grad_sample_flat, grad_i_flat).item())
    return np.array(ntk_values)

def compute_rate_of_change_of_alignment(ntks, losses, lr):
    return lr * np.mean(np.multiply(ntks, losses))

def centroid_statistics(point, centroids):
    centroids(point)
    return {
        'norm': centroids.get_norms()[0].item(),
        'unnormalised': centroids.get_inner_products()[0].item(),
        'normalised': centroids.get_alignments()[0].item()
    }

def train(config):
    run_name=f"{config.width}width-{config.init_scale}initscale-{config.output_scale}outputscale-{config.seed}"
    run = wb.init(project='centroid_alignment_ntk_test', config=vars(config), name=run_name)
    set_global_seed(config.seed)
    loaders = prepare_dataloaders(download_directory=config.download_dir)
    sample_point, _ = next(iter(loaders['train']))
    sample_point = sample_point[:1]

    model = build_model(input_dim=784, config=config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    logged_steps = np.sort(np.unique(np.append(
        np.logspace(0, np.log10(config.steps), config.num_logs, dtype=int), [0, config.steps]
    )))

    for step in tqdm(range(config.steps + 1)):
        config.step = step
        if step in logged_steps:
            wb.log(evaluate(model, loaders, sample_point, config))

        model.train()
        for x, labels in loaders['train']:
            x, labels = x.to(config.device), labels.float().unsqueeze(1).to(config.device)
            optimizer.zero_grad()
            output = config.output_scale * model(x)
            loss = nn.BCEWithLogitsLoss()(output, labels)
            loss.backward()
            optimizer.step()

    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_scale', type=float, default=1.0)
    parser.add_argument('--output_scale', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--num_logs', type=int, default=64)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--download_dir', type=str, default='./data')

    args = parser.parse_args()
    train(args)