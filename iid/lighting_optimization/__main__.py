import hydra
import pytorch_lightning
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from iid.utils import init_logger

import time


def lighting_optimization(cfg: DictConfig):
    module_logger = init_logger("LightingOptimization_MAIN")

    # ============= CONFIG =============
    OmegaConf.resolve(cfg)
    module_logger.info(f"Experiment config: \n{OmegaConf.to_yaml(cfg)}")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        pytorch_lightning.seed_everything(cfg.seed, workers=True)

    device = cfg.device
    if device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
    module_logger.info(f"Running on {device}")

    # ============= DATA =============
    # Use datamodule
    datamodule_cfg = cfg.data
    module_logger.info(f"Instantiating datamodule <{datamodule_cfg._target_}>")
    datamodule = hydra.utils.instantiate(datamodule_cfg)

    # ============= MODEL =============
    # Init model
    model_cfg = cfg.model
    module_logger.info(f"Instantiating model <{model_cfg._target_}>")
    model = hydra.utils.instantiate(model_cfg)
    model = model.to(device)

    # ============= CALLBACKS =============
    module_logger.info("Instantiating callbacks...")
    callbacks = []
    for _, cb_conf in cfg.get("callbacks").items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            module_logger.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # ============= LOGGER =============
    logger_cfg = cfg.get("logger")
    module_logger.info(f"Instantiating logger <{logger_cfg._target_}>")
    logger = hydra.utils.instantiate(logger_cfg)
    if logger is not None:
        hparams = {
            "datamodule": cfg.get("data"),
            "model": cfg.get("model"),
            "task_name": cfg.get("task_name"),
            "tags": cfg.get("tags"),
            "seed": cfg.get("seed"),
        }
        logger.log_hyperparams(hparams)

    # ============= TRAINER =============
    module_logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # ============= OPTIMIZATION =============
    start_time = time.time()
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    print("Time:", time.time()-start_time)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="stage/lighting_optimization.yaml")
def main(cfg: DictConfig):
    lighting_optimization(cfg)


if __name__ == "__main__":
    main()
