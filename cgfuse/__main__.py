import os
import platform
import importlib
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

import datasets
import transformers
import lightning.pytorch as pl
import lightning.pytorch.callbacks as cb
from aim.pytorch_lightning import AimLogger

from .tasks import Task

# Checkpoints store the tokenizer and the Hydra config; torch>=2.6 refuses them with weights_only=True
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(args: DictConfig) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(args, resolve=True, enum_to_str=True))

    driver = importlib.import_module(f".tasks.{cfg.task_name}", __package__)
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if cfg.debug:
        aim_callback = None
    else:
        aim_callback = AimLogger(repo=cfg.aim_repo, experiment=cfg.experiment_name)
        cfg.trainer["ckpt_path"] = (
            Path(cfg.workdir) / Path("ckpt") / Path(aim_callback.experiment.hash)
        )
    try:
        task: Task = driver.main(cfg)

        cp_target_metric = f"val_{cfg.trainer.cp_metric}"

        checkpoint_callback = cb.ModelCheckpoint(
            # Save checkpoints per experiment using experiment hash created by Aim
            dirpath=(
                cfg.trainer.ckpt_path if "ckpt_path" in cfg.trainer else args.workdir
            ),
            save_top_k=cfg.trainer.save_top_k,
            monitor=cp_target_metric,
            mode=cfg.trainer.cp_mode,
            save_last="link",
        )

        lr_monitor = cb.LearningRateMonitor(logging_interval="step")
        callbacks = [checkpoint_callback, lr_monitor]

        if cfg.trainer.es_patience != -1:
            es_target_metric = f"val_{cfg.trainer.es_metric}"
            earlystopping_callback = cb.EarlyStopping(
                monitor=es_target_metric,
                patience=cfg.trainer.es_patience,
                mode=cfg.trainer.es_mode,
                min_delta=0.01,
            )
            callbacks.append(earlystopping_callback)

        auto_opt = (
            {
                "accumulate_grad_batches": cfg.trainer.accumulate_grad_batches,
                "gradient_clip_val": cfg.trainer.gradient_clip_val,
            }
            if cfg.trainer.automatic_optimization
            else {}
        )

        batch_size = (
            cfg.dataset.train.batch_size
            if "train" in cfg.dataset
            else cfg.dataset.batch_size
        )

        trainer = pl.Trainer(
            max_epochs=cfg.trainer.epoch,
            logger=aim_callback,
            default_root_dir=cfg.workdir,
            callbacks=callbacks,
            accelerator=cfg.trainer.accelerator,
            devices="auto" if cfg.trainer.accelerator == "cpu" else cfg.trainer.device,
            precision=cfg.trainer.precision,
            log_every_n_steps=batch_size if batch_size <= 64 else 64,
            **auto_opt,
        )

        trainer.fit(task)
        if cfg.is_test:
            task.test_ckpt = "best"
            trainer.test(ckpt_path="best")

    except Exception as err:
        import traceback
        if not cfg.debug:
            aim_callback.experiment.log_error(traceback.format_exc())
            aim_callback.finalize()
        raise err


if __name__ == "__main__":
    if platform.system() == "Darwin":
        import multiprocessing

        multiprocessing.set_start_method("fork")
    main()
