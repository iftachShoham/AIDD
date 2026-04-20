"""Training and evaluation"""

import hydra
import os
import numpy as np
import run_train
from utils import utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    ngpus = cfg.ngpus
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        
        work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        # Safely get work_dir - use run.dir if available, otherwise construct from sweep config
        try:
            work_dir = hydra_cfg.runtime.output_dir
        except (AttributeError, KeyError):
            try:
                work_dir = hydra_cfg.run.dir
            except (AttributeError, KeyError):
                work_dir = os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    # Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_cfg = HydraConfig.get()
    # Safely check if we're in a sweep mode
    try:
        job_id = hydra_cfg.job.id
        logger.info(f"Run id: {job_id}")
    except (AttributeError, KeyError, Exception):
        pass

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_train.run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()