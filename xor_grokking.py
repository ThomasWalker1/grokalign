import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb as wb
import argparse
from tqdm import tqdm

from utils import GrokAlign, Centroids

def generate_xor_data(n, p, epsilon):
    x_signal = np.random.choice([-1, 1], size=(n, 2))
    x_noise = np.random.choice([-epsilon, epsilon], size=(n, p - 2))
    X = np.hstack([x_signal, x_noise])
    y = x_signal[:, 0] * x_signal[:, 1]
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)

def centroid_statistics(point, centroids):
    centroids(point)
    inner_product = centroids.get_inner_products()[0].item()
    norm = centroids.get_norms()[0].item()
    return {
        'centroid_norm': norm,
        'centroid_alignment': inner_product / torch.clamp(norm * point.norm(), min=1e-8).item()
    }

def evaluate(model, X, y):
    with torch.no_grad():
        preds = torch.sign(model(X).squeeze())
        accuracy = (preds == y).float().mean().item()
    return accuracy

def evaluate_perturbations(model, X_test, y_test, amplitudes):
    results = {}
    perturbation = torch.randn_like(X_test[:, 2:])
    for amp in amplitudes:
        perturbed = X_test.clone()
        perturbed[:, 2:] += perturbation * amp
        acc = evaluate(model, perturbed, y_test)
        results[f'{amp}_accuracy'] = acc
    return results

def train(config, device='cuda'):
    run_name = f"{config.p}p-{config.n}n-{config.epsilon}epsilon-{config.hdim}hdim-{config.weight_decay}Wd-{config.lambda_jac}GA"
    run = wb.init(project='xor_grokking', config=config, name=run_name)

    torch.manual_seed(0)
    np.random.seed(0)

    X, y = generate_xor_data(config.n, config.p, config.epsilon)
    X_test, y_test = generate_xor_data(config.n, config.p, config.epsilon)
    X, y, X_test, y_test = X.to(device), y.to(device), X_test.to(device), y_test.to(device)
    point = X[:1]

    model = nn.Sequential(
        nn.Linear(config.p, config.hdim, bias=False),
        nn.ReLU(),
        nn.Linear(config.hdim, 1, bias=False)
    ).to(device).type(torch.float64)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    grokalign = GrokAlign(model) if config.lambda_jac > 0.0 else None
    centroids = Centroids(model)

    logged_steps = np.unique(np.append(np.logspace(0, np.log10(config.steps), config.n_logs, dtype=int), [0, config.steps]))
    pbar = tqdm(range(config.steps + 1))

    for step in pbar:
        if step in logged_steps:
            model.eval()
            stats = {
                'step': step,
                'train_accuracy': evaluate(model, X, y),
                'test_accuracy': evaluate(model, X_test, y_test),
                **evaluate_perturbations(model, X_test, y_test, amplitudes=[0.2, 0.4, 0.6, 0.8, 1.0]),
                **centroid_statistics(point, centroids)
            }
            wb.log(stats)

        model.train()
        optimizer.zero_grad()
        output = model(X).squeeze()
        loss = F.mse_loss(output, y)
        if grokalign is not None:
            loss += config.lambda_jac * grokalign(X)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'{step} - Loss: {loss.item():.4f}')

    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=int, default=40_000)
    parser.add_argument('--n', type=int, default=400)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--hdim', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--n_logs', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    train(args)