import datetime
import os
import os.path
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gc
from itertools import chain

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from data import data
from core.losses import losses
from core.samplers import sampling
from core.utils import graph_lib
from core.utils import noise_lib
from utils import utils
from core.models import SEDD
from core.models.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import wandb  # Import wandb


torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()


def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    mprint(f"noise: {noise}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0)

    mprint("state built.")

    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    mprint("state restored.")
    initial_step = int(state['step'])
    mprint(f"initial step: {initial_step}")

    mprint("getting dataloaders...")
    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)


    mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(
        noise, graph, True, optimize_fn,
        cfg.training.accum,
        lamda=cfg.training.lamda,
        mask_token_id=cfg.tokens,
    )
    eval_step_fn = losses.get_step_fn(
        noise, graph, False, optimize_fn,
        cfg.training.accum,
        lamda=cfg.training.lamda,
        mask_token_id=cfg.tokens,
    )


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")
    
    # Initialize WandB for logging
    if rank == 0:
        wandb.init(
        project=cfg.get("wandb_project", "Discrete-Diffusion-Inpainting"),
        entity=cfg.get("wandb_entity", None),
        config={
            "learning_rate": cfg.optim.lr,
            "weight_decay": cfg.optim.weight_decay,
            "optimizer": cfg.optim.optimizer,
            "beta1": cfg.optim.beta1,
            "beta2": cfg.optim.beta2,
            "eps": cfg.optim.eps,
            "warmup": cfg.optim.warmup,
            "grad_clip": cfg.optim.grad_clip,
            "batch_size": cfg.training.batch_size,
            "work_dir": work_dir,
        },
        tags=["absorbing", "musicnet", "not_masked_train", "fo_diff_loss"]
)

    best_val_loss = float('inf')
    patience = cfg.training.patience
    patience_counter = 0

    while state['step'] < num_train_steps + 1:
        step = state['step']


        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)
        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
                
                if rank == 0:  # Only log from rank 0 to avoid duplicate logging
                    wandb.log({"training_loss": loss.item(), "step": step})
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            # valid step
            if step % cfg.training.eval_freq == 0 and cfg.data.valid == "Maestro":
                
                eval_batch = next(eval_iter)['input_ids'].to(device)
                eval_loss = eval_step_fn(state, eval_batch)
            
                dist.all_reduce(eval_loss)
                eval_loss /= world_size
            
                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))
                wandb.log({"validation_loss": eval_loss.item(), "step": step})

                # ---------- EARLY STOPPING ----------
                if eval_loss.item() < best_val_loss:
                    best_val_loss = eval_loss.item()
                    patience_counter = 0

                    # Save the real best model so far
                    if rank == 0:
                        utils.save_checkpoint(
                            os.path.join(checkpoint_dir, f'best_model.pth'), state
                        )
                        mprint(f"New best model saved with val_loss={best_val_loss:.5e} at step {step}.")

                else:
                    patience_counter += 1
                    mprint(f"Validation did not improve. Patience {patience_counter}/{patience}.")

                if patience_counter >= patience:
                    mprint("Early stopping triggered. Stopping training...")
                    break
                # ------------------------------------

            # end valid step

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)


