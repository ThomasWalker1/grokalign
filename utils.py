import torch
import torch.nn as nn

class GrokAlign:
    """
    Reference: https://arxiv.org/abs/1908.02729
    """
    def __init__(self, model: nn.Module, num_projections: int = 1, device=None):
        self.model = model
        self.num_projections = num_projections
        self.device = device if device is not None else next(model.parameters()).device

    def _get_random_projections(self, batch_size, output_dim):
        v = torch.randn(self.num_projections, batch_size, output_dim, device=self.device)
        return v / torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-8)

    def compute_jacobian_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)

        output = self.model(x)
        batch_size, output_dim = output.shape

        v_vectors = self._get_random_projections(batch_size, output_dim)

        norm_sum = torch.zeros(batch_size, device=self.device)

        for i in range(self.num_projections):
            v = v_vectors[i]
            Jv = torch.autograd.grad(
                outputs=output,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )[0]

            Jv_flat = Jv.flatten(start_dim=1)
            norm_sum += (Jv_flat ** 2).sum(dim=1)

        scaling = output_dim / self.num_projections
        return torch.sqrt(torch.clamp(norm_sum * scaling, min=1e-8))

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.compute_jacobian_norm(inputs).mean()

class Centroids:
    def __init__(self, model: nn.Module, device=None):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device
        self.centroids = None
        self.inner_products = None
        self.norms = None
        self.alignments = None

    def __call__(self, x: torch.Tensor):
        x = x.requires_grad_(True).to(self.device)
        self.centroids = self._compute_centroids(x)
        flat_centroids = self.centroids.reshape(self.centroids.size(0), -1)
        flat_x = x.reshape(x.size(0), -1)

        self.inner_products = (flat_centroids * flat_x).sum(dim=1)

        self.alignments = self.inner_products / torch.clamp(
            flat_centroids.norm(dim=1) * flat_x.norm(dim=1), min=1e-8
        )

        self.norms = flat_centroids.norm(dim=1)

        return self

    def _compute_centroids(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        grad_output = torch.ones_like(output)
        centroids = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True
        )[0]
        return centroids

    def get_centroids(self) -> torch.Tensor:
        return self.centroids
    
    def get_inner_products(self) -> torch.Tensor:
        return self.inner_products

    def get_norms(self) -> torch.Tensor:
        return self.norms
    
    def get_alignments(self) -> torch.Tensor:
        return self.alignments
    
class PC1:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def _compute_pc1s(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True).to(self.device)
        pc1s=[]
        for k in range(x.size()[0]):
            J=torch.autograd.functional.jacobian(self.model,x[k:k+1])[0,:,0]
            J=J.reshape(J.size()[0],-1)
            U, S, Vh = torch.linalg.svd(J, full_matrices=False)
            S_squared = S ** 2
            explained_var = S_squared[0] / S_squared.sum()
            pc1s.append(explained_var.item())
        pc1s=torch.tensor(pc1s)
        return pc1s

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pc1s = self._compute_pc1s(x)
        return pc1s.mean().item()
    

## Grokfast, https://arxiv.org/abs/2405.20233

from collections import deque
from typing import Dict, Optional, Literal


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False, # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach()) # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads

## Transformer Alignment, Reference: https://github.com/mechanistic-interpretability-grokking

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from tqdm import tqdm

class HookPoint(nn.Module):
    '''A helper class to get access to intermediate activations (inspired by Garcon)
    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient way to add PyTorch hooks
    '''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x
    
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model, fixed=False):
        super().__init__()
        if fixed:
            self.W_E = torch.randn(d_model, d_vocab)/np.sqrt(d_model)
        else:
            self.W_E = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_model))
    
    def forward(self, x):
        W_E = self.W_E.to(x.device)
        return torch.einsum('dbp -> bpd', W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model, fixed=False):
        super().__init__()
        if fixed:
            self.W_U = torch.randn(d_model, d_vocab)/np.sqrt(d_vocab)
        else:
            self.W_U = nn.Parameter(torch.randn(d_model, d_vocab)/np.sqrt(d_vocab))
    
    def forward(self, x):
        W_U = self.W_U.to(x.device)
        return (x @ W_U)

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)/np.sqrt(d_model))
    
    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]
        
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked/np.sqrt(self.d_head), dim=-1)
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        x = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return x
    
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))

        self.hook_post = HookPoint()
        
    def forward(self, x):
        x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        x = self.hook_post(F.relu(x))
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp)
    
    def forward(self, x):

        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x
    
class Transformer(nn.Module):
    def __init__(self, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, fixed=False):
        super().__init__()

        self.embed = Embed(d_vocab, d_model, fixed=fixed)
        self.pos_embed = PosEmbed(n_ctx, d_model)
        self.block = TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx)
        self.unembed = Unembed(d_vocab, d_model, fixed=fixed)

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        x = self.block(x)
        x = self.unembed(x)
        return x
    
    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
    
    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')
    
    def cache_all(self, cache, incl_bwd=False):
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')


def full_loss(model, loader, device):
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[:,-1]
    return torch.nn.functional.cross_entropy(logits, labels)

def full_accuracy(model, loader, device):
    x, labels = next(iter(loader))
    x = x.to(device)
    labels = labels.to(device)
    logits = model(x)[:,-1]
    predictions = torch.argmax(logits, dim=1)
    return torch.sum(predictions == labels).item() / len(labels)

def cross_entropy_high_precision(logits, labels):
    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def test_logits(logits, p, is_train, is_test, labels, bias_correction=False, original_logits=None, mode='all'):
    if logits.shape[1]==p*p:
        logits = logits.T
    if logits.shape==torch.Size([p*p, p+1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p*p, p)
    if bias_correction:
        logits = einops.reduce(original_logits - logits, 'batch ... -> ...', 'mean') + logits
    if mode=='train':
        return cross_entropy_high_precision(logits[is_train], labels[is_train])
    elif mode=='test':
        return cross_entropy_high_precision(logits[is_test], labels[is_test])
    elif mode=='all':
        return cross_entropy_high_precision(logits, labels)

def fourier_2d_basis_term(x_index, y_index, fourier_basis):
    return (fourier_basis[x_index][:, None] * fourier_basis[y_index][None, :]).flatten()

def get_component_cos_xpy(tensor, freq, fourier_basis, collapse_dim=False):
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1, fourier_basis=fourier_basis).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq, fourier_basis=fourier_basis).flatten()
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, fourier_basis, collapse_dim=False):
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1, fourier_basis=fourier_basis).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq, fourier_basis=fourier_basis).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)

def calculate_excluded_loss(config, fourier_basis, key_freqs, is_train, is_test, labels, logits):
    row = []
    for freq in key_freqs:
        cos = get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis) 
        sin = get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis) 
        value = test_logits(logits - cos - sin, bias_correction=False, mode='train', p = config.p,
           is_train = is_train, is_test = is_test, labels = labels)
        row.append(value.item())
    row=torch.tensor(row)
    return row.mean().item()

def get_components_of_trig_loss(logits, freq, fourier_basis):
    cos = get_component_cos_xpy(logits, freq, fourier_basis=fourier_basis)
    sin = get_component_sin_xpy(logits, freq, fourier_basis=fourier_basis)
    return cos + sin

def calculate_trig_loss(config, model, train, logits, key_freqs, fourier_basis, all_data, is_train, is_test, labels, mode='all'):
    trig_logits = sum([get_components_of_trig_loss(logits, freq, fourier_basis) for freq in key_freqs])
    return test_logits(trig_logits,p = config.p,is_train = is_train,is_test = is_test,labels = labels,bias_correction=True,original_logits=logits,mode=mode)

def calculate_coefficients(logits, fourier_basis, key_freqs, p, device):
    '''updated version from https://colab.research.google.com/drive/1ScVRL8OCtTFpOHpgfz0PLTFvX4g_YbuN?usp=sharing#scrollTo=WY4nPUDwl9UN
    '''
    x = torch.arange(p)[None, :, None, None]
    y = torch.arange(p)[None, None, :, None]
    z = torch.arange(p)[None, None, None, :]
    w = torch.arange(1, (p//2+1))[:, None, None, None]
    coses = torch.cos(w*torch.pi*2/p * (x + y - z)).to(device)
    coses = coses.reshape(p//2, p*p, p)
    coses/= coses.pow(2).sum([-2, -1], keepdim=True).sqrt()
    cos_coefficients = (coses * logits).sum([-2, -1])
    return cos_coefficients.mean().item()

def fft2d(mat, p, fourier_basis):
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis, fourier_basis)
    return fourier_mat.reshape(shape)

def make_fourier_basis(config,device):
    fourier_basis = []
    fourier_basis.append(torch.ones(config.p)/np.sqrt(config.p))
    fourier_basis_names = ['Const']
    for i in range(1, config.p//2 +1):
        fourier_basis.append(torch.cos(2*torch.pi*torch.arange(config.p)*i/config.p))
        fourier_basis.append(torch.sin(2*torch.pi*torch.arange(config.p)*i/config.p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    return torch.stack(fourier_basis, dim=0).to(device)

def unflatten_first(tensor, p):
    if tensor.shape[0]==p*p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else: 
        return tensor

def extract_freq_2d(tensor, freq, p):
    tensor = unflatten_first(tensor, p)
    index_1d = [0, 2*freq-1, 2*freq]
    return tensor[[[i]*3 for i in index_1d], [index_1d]*3]

def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def calculate_key_freqs(config, model, all_data, device):

    cache = {}
    model.remove_all_hooks() # TODO is this line fucky??
    model.cache_all(cache)
    model(all_data)
    neuron_acts = cache['block.mlp.hook_post'][:, -1]

    neuron_acts_centered = neuron_acts - einops.reduce(neuron_acts, 'batch neuron -> 1 neuron', 'mean')
    fourier_basis = make_fourier_basis(config = config, device=device)
    fourier_neuron_acts = fft2d(neuron_acts_centered, p = config.p, fourier_basis=fourier_basis)

    fourier_neuron_acts_square = fourier_neuron_acts.reshape(config.p, config.p, config.d_mlp)
    neuron_freqs = []
    neuron_frac_explained = []
    for ni in range(config.d_mlp):
        best_frac_explained = -1e6
        best_freq = -1
        for freq in range(1, config.p//2):
            numerator = extract_freq_2d(fourier_neuron_acts_square[:, :, ni], freq, p = config.p).pow(2).sum()
            denominator = fourier_neuron_acts_square[:, :, ni].pow(2).sum().item()
            frac_explained = numerator / denominator
            if frac_explained > best_frac_explained:
                best_freq = freq
                best_frac_explained = frac_explained
        neuron_freqs.append(best_freq)
        neuron_frac_explained.append(best_frac_explained)
    neuron_freqs = np.array(neuron_freqs)
    neuron_frac_explained = to_numpy(neuron_frac_explained)
    key_freqs, neuron_freq_counts = np.unique(neuron_freqs, return_counts=True)
    return key_freqs

def is_train_is_test(config,train):
    train=[(i[0].item(),i[1].item(),i[2].item()) for (i, j) in train]
    is_train = []
    is_test = []
    for x in tqdm(range(config.p)):
        for y in range(config.p):
            if (x, y, 113) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)
    is_train = np.array(is_train)
    is_test = np.array(is_test)
    return (is_train, is_test)

def sum_sq_weights(model):
    row = []
    for name, param in model.named_parameters():
        row.append(param.pow(2).sum().item())
    return sum(row)

def gini(x: torch.Tensor) -> float:
    x = x.flatten()
    if x.numel() == 0:
        return 0.0

    x_sorted, _ = torch.sort(torch.abs(x))
    n = x.numel()
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)

    gini_coeff = (2 * torch.sum(index * x_sorted)) / (n * torch.sum(x_sorted)) - (n + 1) / n
    return gini_coeff.item()


def gini_from_fourier_norms(matrix: torch.Tensor) -> float:
    matrix_dft = torch.fft.fft2(matrix, dim=1)
    norms = torch.linalg.norm(matrix_dft, dim=0)
    return gini(norms)