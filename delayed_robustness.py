import torch
import torch.nn as nn
import torchvision
import einops

from ffcv.loader import Loader, OrderOption
from ffcv.fields import IntField, RGBImageField
from ffcv.transforms import ToTensor, ToDevice, Convert, ToTorchImage
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from autoattack import AutoAttack # Our utilisation of AutoAttack is slighlty modified to output the desired metric, reference https://github.com/fra31/auto-attack

import os
import wandb as wb
from tqdm import tqdm
import numpy as np
import argparse

from utils import GrokAlign, Centroids, PC1

def construct_pipeline(config):
    image_pipeline = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(torch.device(config.device), non_blocking=True),
        ToTorchImage(),
        Convert(torch.float32),
        torchvision.transforms.Normalize(
            np.array([0.4914, 0.4822, 0.4465]) * 255,
            np.array([0.2471, 0.2435, 0.2616]) * 255
        )
    ]
    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(torch.device(config.device)),
        Squeeze()
    ]
    return {'image': image_pipeline, 'label': label_pipeline}

def write_beton_files(dataset, filename, download_dir):
    os.makedirs(os.path.join(download_dir, 'beton'), exist_ok=True)
    path = os.path.join(download_dir, 'beton', f'{filename}.beton')
    writer = DatasetWriter(path, {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(dataset)

def construct_dataloaders(config):
    train_ds = torchvision.datasets.CIFAR10(train=True, download=True, root=config.download_dir)
    train_ds = torch.utils.data.Subset(train_ds, range(1024))
    test_ds = torchvision.datasets.CIFAR10(train=False, download=True, root=config.download_dir)
    test_ds = torch.utils.data.Subset(test_ds, range(1024))

    write_beton_files(train_ds, 'cifar10_train_dataset', config.download_dir)
    write_beton_files(test_ds, 'cifar10_test_dataset', config.download_dir)

    train_path = os.path.join(config.download_dir, 'beton', 'cifar10_train_dataset.beton')
    test_path = os.path.join(config.download_dir, 'beton', 'cifar10_test_dataset.beton')

    train_loader = Loader(
        train_path,
        batch_size=config.batch_size,
        num_workers=16,
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines=construct_pipeline(config)
    )

    test_loader = Loader(
        test_path,
        batch_size=config.batch_size,
        num_workers=16,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        pipelines=construct_pipeline(config)
    )

    return train_loader, test_loader

def compute_accuracy(model, loader, device):
    correct = total = 0
    with torch.no_grad():
        for x, labels in loader:
            preds = torch.argmax(model(x.to(device)), dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += x.size(0)
    return correct / total

def construct_model(config):
    model = nn.Sequential(
        nn.Conv2d(3, config.filt, 3, padding=1, bias=False), nn.ReLU(),
        nn.Conv2d(config.filt, config.filt, 3, stride=2, padding=1, bias=False), nn.ReLU(),
        nn.Conv2d(config.filt, config.filt*2, 3, padding=1, bias=False), nn.ReLU(),
        nn.Conv2d(config.filt*2, config.filt*2, 3, stride=2, padding=1, bias=False), nn.ReLU(),
        nn.Conv2d(config.filt*2, config.filt*4, 3, padding=1, bias=False), nn.ReLU(),
        nn.Conv2d(config.filt*4, config.filt*4, 3, stride=2, padding=1, bias=False), nn.ReLU(),
        nn.AvgPool2d(2, 2), nn.AvgPool2d(2, 2), nn.Flatten(),
        nn.Linear(config.filt*4, 128, bias=False), nn.ReLU(),
        nn.Linear(128, 10, bias=False)
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return model

def train(config):
    run_name = f"{config.loss_fn}-{config.weight_decay}Wd-{config.lambda_jac}GA"
    wb.init(project='delayed_robustness', config=config, name=run_name)
    torch.manual_seed(config.seed)
    if 'cuda' in config.device:
        torch.cuda.manual_seed_all(config.seed)

    train_loader, test_loader = construct_dataloaders(config)
    model = construct_model(config).to(config.device)
    one_hots = torch.eye(10).to(config.device)

    grokalign = GrokAlign(model) if config.lambda_jac > 0 else None
    centroids = Centroids(model)
    pc1 = PC1(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    logged_steps = np.unique(np.append(np.logspace(0, np.log10(config.steps), config.num_logs, dtype=int), [0, config.steps]))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    ])
    adv_ds = torchvision.datasets.CIFAR10(train=False, download=True, transform=transform, root=config.download_dir)
    adv_ds = torch.utils.data.Subset(adv_ds, range(1024))
    adv_loader = torch.utils.data.DataLoader(adv_ds, batch_size=256)
    x_test = torch.cat([x for x, _ in adv_loader], 0).to(config.device)
    y_test = torch.cat([y for _, y in adv_loader], 0).to(config.device)

    for step in tqdm(range(config.steps + 1)):
        model.train()
        for x, labels in train_loader:
            x, labels = x.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            output = model(x)
            if config.loss_fn == 'MSE':
                loss = nn.MSELoss()(output, one_hots[labels])
            elif config.loss_fn == 'CrossEntropy':
                loss = nn.CrossEntropyLoss()(output, labels)
            if grokalign:
                loss += config.lambda_jac * grokalign(x)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

        if step in logged_steps:
            model.eval()
            stats = {'step': step}
            with torch.no_grad():
                for name, loader in {'train': train_loader, 'test': test_loader}.items():
                    stats[f'{name}_accuracy'] = compute_accuracy(model, loader, config.device)
            adversary = AutoAttack(model, norm='Linf', eps=4/255., verbose=False, attacks_to_run=['apgd-ce'], version='custom')
            stats['test_adv_accuracy_4'] = adversary.run_standard_evaluation_individual(x_test, y_test, bs=256)['apgd-ce']
            stats['centroid_alignment'] = centroids(x).get_alignments().mean().item()
            stats['pc1'] = pc1(x[:64])
            wb.log(stats)

    torch.save(model.state_dict(), f'{config.model_dir}/model.pth')
    artifact = wb.Artifact(name=f'{run_name}_{step}', type='model')
    artifact.add_file(f'{config.model_dir}/model.pth')
    wb.log_artifact(artifact)
    wb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_fn', type=str, default='MSE', choices=['MSE', 'CrossEntropy'])
    parser.add_argument('--filt', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lambda_jac', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--steps', type=int, default=36000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_logs', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--download_dir', type=str, default='./data')
    parser.add_argument('--model_dir', type=str, default='./outputs')
    args = parser.parse_args()
    train(args)
