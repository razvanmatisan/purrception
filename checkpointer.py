# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/wav2vec2/common/helpers.py

import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
from torch_ema import ExponentialMovingAverage


class Checkpointer:
    def __init__(
        self,
        log_path: str,
        exp_name: str,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        dataset: str,
        backbone_type: str,
        resume: bool,
    ):
        self.log_path = Path(log_path) / exp_name

        if save_checkpoint_path:
            pattern = f"{save_checkpoint_path}_step-*"
            self.save_path = f"{save_checkpoint_path}"
        else:
            pattern = f"{dataset}_{backbone_type}_step-*.pth"
            self.save_path = f"{dataset}_{backbone_type}.pth"

        self.save_path = self.log_path / self.save_path

        checkpoint_last = self.save_path.parent / (self.save_path.stem + "_last" + self.save_path.suffix)
        self.checkpoint_last = checkpoint_last if checkpoint_last.is_file() else None

        tracked = [(re.search("step-(\d+)\.pth", str(f)).group(1), f) for f in Path(self.log_path).rglob(pattern)]

        self.tracked = self.tracked = OrderedDict(sorted(tracked, key=lambda t: t[0]))

        fpath = load_checkpoint_path or (self.last_checkpoint() if resume else None)

        if fpath is not None:
            print(f"Loading backbone from {fpath}")
            self.last_state = torch.load(fpath, map_location="cpu", weights_only=False)
        else:
            self.last_state = None

    def save(
        self,
        backbone: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        ema: ExponentialMovingAverage,
        logs,
        step: int,
        new_checkpoint: bool,
    ):
        path = self.save_path

        if new_checkpoint:
            path = path.parent / (path.stem + f"_step-{step}" + path.suffix)
        else:
            path = path.parent / (path.stem + "_last" + path.suffix)

        state = {
            "step": step,
            "backbone_state_dict": backbone.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
            "ema_state_dict": ema.state_dict() if ema else None,
            "train_state": {"logs": logs, "step": step},
        }

        self.last_state = state

        print(f"Saving {path}...")
        torch.save(state, path)

    def _is_deepspeed_optimizer(self, checkpoint_path):
        """Check if optimizer is a DeepSpeed optimizer"""
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            state = self.last_state

        if "optimizer_state_dict" in state:
            optimizer_state = state["optimizer_state_dict"]
            print(f"Optimizer state keys: {list(optimizer_state.keys())}")

            if "zero_stage" in optimizer_state:
                print("Detected DeepSpeed optimizer - using DeepSpeed loading method")
                return True

        return False

    def maybe_load_state(
        self,
        checkpoint_path=None,
        backbone: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        ema: ExponentialMovingAverage = None,
        train_state: Dict = None,
    ):
        if checkpoint_path is None and self.last_state is None:
            print("No checkpoint to load")
            return

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            state = self.last_state

        if backbone is not None:
            if "backbone_state_dict" in state:
                backbone.load_state_dict(state["backbone_state_dict"])
            else:
                backbone.load_state_dict(state)

        if ema is not None:
            if state.get("ema_state_dict") is not None:
                print("Loading EMA state from checkpoint...")
                ema.load_state_dict(state["ema_state_dict"])
            else:
                # This allows loading older checkpoints that didn't have EMA saved
                print("WARNING: EMA state not found in checkpoint. The EMA model will be initialized from scratch.")

        if lr_scheduler is not None:
            if "lr_scheduler_state_dict" in state:
                try:
                    lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
                    print("Successfully loaded LR scheduler state.")
                except Exception as e:
                    print(f"Failed to load LR scheduler state: {e}")
                    print("Continuing without loading LR scheduler state.")
            else:
                print("WARNING: LR scheduler state not found in checkpoint.")

        if train_state is not None:
            if "train_state" in state:
                try:
                    train_state.update(state["train_state"])
                    print("Successfully loaded training state.")
                except Exception as e:
                    print(f"Failed to load training state: {e}")
                    print("Continuing without loading training state.")
            else:
                print("WARNING: Training state not found in checkpoint.")

        if optimizer is not None:
            if "optimizer_state_dict" in state:
                try:
                    optimizer.load_state_dict(state["optimizer_state_dict"])
                    print("Successfully loaded optimizer state.")
                except Exception as e:
                    print(f"Failed to load optimizer state: {e}")
                    print("Continuing without loading optimizer state.")
            else:
                print("WARNING: Optimizer state not found in checkpoint.")

        return True

    def last_checkpoint(self):
        tracked = list(self.tracked.values())
        if self.checkpoint_last is not None:
            tracked += [self.checkpoint_last]

        for fpath in reversed(tracked):
            try:
                torch.load(fpath, map_location="cpu", weights_only=False)
                print(f"Checkpoint {fpath} loaded successfully.")
                return fpath
            except (IOError, OSError, RuntimeError) as e:
                print(f"Checkpoint {fpath} appears corrupted: {e}")

        return None
