import os
import torch
from core.models import SEDD
from utils.utils import load_hydra_config_from_run
from core.models.ema import ExponentialMovingAverage
import core.utils.graph_lib
import core.utils.noise_lib
from pathlib import Path

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    root = Path(root_dir)

    if root.suffix == ".pth":
        # root is checkpoint file
        ckpt_path = root
        run_dir = root.parent.parent  # go up: .../checkpoints/checkpoint_x.pth -> run dir
    else:
        # root is a run directory
        run_dir = root
        ckpt_path = run_dir / "checkpoints-meta" / "checkpoint.pth"

    # load Hydra config from run directory
    cfg = load_hydra_config_from_run(str(run_dir))
    graph = core.utils.graph_lib.get_graph(cfg, device)
    noise = core.utils.noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    print(f"ckpt_path: {ckpt_path}")
    loaded_state = torch.load(ckpt_path, map_location=device)

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model_from_checkpoint(ckpt_path, device):
    """
    Load a model from a checkpoint file and a config.yaml in the same directory.

    Expected layout:
        <dir>/checkpoint.pth
        <dir>/config.yaml
    """
    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {ckpt_path.parent}")

    cfg = OmegaConf.load(config_path)
    graph = core.utils.graph_lib.get_graph(cfg, device)
    noise = core.utils.noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    try:
        return load_model_hf(root_dir, device)
    except:
        return load_model_local(root_dir, device)