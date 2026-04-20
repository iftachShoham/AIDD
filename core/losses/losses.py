import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from core.utils import graph_lib
from core.models import utils as mutils




def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, lamda=0.0, mask_token_id=None):

    def loss_fn(model, batch, cond=None, t=None, perturbed_batch=None):
        """Compute the score-entropy loss for a batch of token sequences.

        Args:
            model: Score model (possibly DDP-wrapped).
            batch: Clean token indices, shape [B, L].
            cond: Optional conditioning input (unused).
            t: Optional noise levels, shape [B]. Sampled uniformly if None.
            perturbed_batch: Optional pre-corrupted tokens. Sampled from the
                forward process if None.

        Returns:
            Per-sample loss tensor, shape [B].
        """
        if t is None:
            if lv:
                raise NotImplementedError("Learned variance sampling is not yet implemented.")
            t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps

        sigma, dsigma = noise(t)

        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma)

        base_model = model.module if hasattr(model, "module") else model
        deriv_reg = compute_deriv_reg(log_score, batch, perturbed_batch, base_model.vocab_embed, mask_token_id)

        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
        loss = (dsigma[:, None] * loss).sum(dim=-1)
        loss = loss + lamda * deriv_reg

        return loss

    return loss_fn
    
    
def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum, lamda=0.0, mask_token_id=None):
    loss_fn = get_loss_fn(noise, graph, train, lamda=lamda, mask_token_id=mask_token_id)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
        
            loss = loss_fn(model, batch, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn

def compute_deriv_reg(log_score, gt, perturbed_batch, vocab_embed, mask_token_id):
    gt_emb = vocab_embed(gt)
    gt_diff = gt_emb[:, 1:] - gt_emb[:, :-1]
    probs = F.softmax(log_score, dim=-1)
    pred_emb = probs @ vocab_embed.embedding.to(probs.dtype)
    pred_diff = pred_emb[:, 1:] - pred_emb[:, :-1]
    mask = (perturbed_batch[:, 1:] == mask_token_id) | (perturbed_batch[:, :-1] == mask_token_id)
    mask = mask.unsqueeze(-1)
    diffs = (pred_diff - gt_diff) * mask
    return (diffs ** 2).sum() / mask.sum().clamp(min=1)