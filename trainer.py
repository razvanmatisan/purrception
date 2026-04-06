import contextlib
import json
import os
import sys
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage
from tqdm import trange

import wandb
from checkpointer import Checkpointer
from latent_diffusion.ldm.util import instantiate_from_config
from models.dit import DiT
from utils.dataset_utils import SimpleImageDataset
from utils.sample_utils import sample_cfm, sample_cfm_endpoint, sample_dfm, sample_p, sample_purrception

# Add LlamaGen to path for llamagen autoencoder support
sys.path.insert(0, os.path.abspath("./LlamaGen"))
try:
    from tokenizer.tokenizer_image.vq_model import VQ_models

    LLAMAGEN_AVAILABLE = True
except ImportError:
    LLAMAGEN_AVAILABLE = False


class Trainer:
    def __init__(
        self,
        model_type: str,
        backbone_type: str,
        backbone_hparams: dict,
        autoencoder_hparams: dict,
        num_iterations: int,
        batch_size: int,
        use_amp: bool,
        amp_dtype: str,
        max_grad_norm: float,
        optimizer_type: str,
        lr: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
        num_warmup_steps: int,
        cfg_scale: float,
        log_path_dir: str,
        exp_name: str,
        eval_every_n_steps: int,
        load_checkpoint_path: str,
        save_checkpoint_path: str,
        save_every_n_steps: int,
        save_new_every_n_steps: int,
        resume: bool,
        ode_steps: int,
        ode_method: str,
        num_samples: int,
        gradient_accumulation_steps: int,
        project_name: str,
        num_workers: int,
        data_fraction: float,
        temperature_softmax: bool,
        decay_ema: float,
        use_ema: bool,
        use_log_z: bool,
        lambda_z: float,
        wandb_run_id: str,
        dataset: str = "imagenet",
        ds_config_path: str = "zero2.json",
    ):
        super(Trainer, self).__init__()

        # Initialize the parameters
        params = locals()
        for key, value in params.items():
            if key != "self":
                setattr(self, key, value)

        # Initialize device
        self._init_deepspeed()

        # Initialize logging stuff
        self._make_log_dir(log_path_dir, exp_name)

        # Initialize amp (dtype only, accelerator handles casting)
        self._init_amp()

        # Initialize checkpointer
        self._init_checkpointer()

        # Initialize backbone and vq model
        self._prepare_autoencoder()
        self._init_backbone()
        self.ema = ExponentialMovingAverage(self.backbone.parameters(), decay=self.decay_ema) if self.use_ema else None

        # Initialize dataloader
        self._init_dataloader()

        # Initialize optimizer and LR scheduler
        self._init_optimizer()
        self._init_lr_scheduler()

        # Save hyperparameters in a JSON file
        self._save_hparams()

        # Initialize wandb instance
        self._init_wandb()

        if self.is_main_process():
            print(f"Gradient clipping enabled with max_grad_norm: {self.max_grad_norm}")

        # Define the sampling function based on the model type
        if self.model_type == "purrception":
            self.sample = sample_purrception
        elif self.model_type == "cfm":
            self.sample = sample_cfm
        elif self.model_type == "cfm-endpoint":
            self.sample = sample_cfm_endpoint
        elif self.model_type == "dfm":
            self.sample = sample_dfm
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Prepare instances for accelerator
        self._prepare_models()

        # Test LlamaGen encoding/decoding if using llamagen autoencoder
        if self.autoencoder_hparams["autoencoder_type"] == "llamagen" and self.is_main_process():
            self._test_llamagen_encoding_decoding()

        # Adaptive gradient clipping will be handled in the training loop

    def _make_log_dir(self, log_path: str, exp_name: str):
        self.log_path = Path(log_path) / exp_name
        if self.is_main_process():
            print(f"Log directory is {self.log_path}")
        self.log_path.mkdir(parents=True, exist_ok=True)

    def _init_wandb(self):
        """Initialize wandb with support for resuming runs"""
        if self.is_main_process():
            print(self.project_name)
            # Priority order for determining run_id:
            # 1. Manually provided wandb_run_id parameter (highest priority)
            # 2. Saved wandb_state.json file (for auto-resume)
            # 3. Create new run (lowest priority)

            wandb_state_path = self.log_path / "wandb_state.json"
            resume_run_id = None

            # Check if resuming and run_id was manually provided
            if self.resume and self.wandb_run_id:
                resume_run_id = self.wandb_run_id
                print(f"Using manually provided wandb run ID: {resume_run_id}")

            # If resuming and no manual run_id, try to load from saved state
            elif self.resume and wandb_state_path.exists():
                try:
                    with open(wandb_state_path, "r") as f:
                        wandb_state = json.load(f)
                        resume_run_id = wandb_state.get("run_id")
                        if resume_run_id:
                            print(f"Found saved wandb run ID: {resume_run_id}")
                except Exception as e:
                    print(f"Could not load wandb state: {e}")

            print(f"resume_run_id: {resume_run_id}")

            # Initialize wandb run
            if resume_run_id:
                # Resume existing run
                try:
                    self.run = wandb.init(
                        project=self.project_name,
                        name=self.exp_name,
                        id=resume_run_id,
                        resume="must",  # Must resume the exact run
                        config=vars(self),
                    )
                    self.wandb_run_id = self.run.id
                    print(f"Successfully resumed wandb run: {self.wandb_run_id}")

                    # Save the run_id for future resuming (in case it wasn't saved before)
                    wandb_state = {"run_id": self.wandb_run_id}
                    with open(wandb_state_path, "w") as f:
                        json.dump(wandb_state, f)

                except Exception as e:
                    print(f"Failed to resume wandb run {resume_run_id}: {e}")
                    print("Creating a new wandb run instead...")
                    self.run = wandb.init(project=self.project_name, name=self.exp_name, config=vars(self))
                    self.wandb_run_id = self.run.id
                    print(f"Created new wandb run: {self.wandb_run_id}")
            else:
                # Create new run
                self.run = wandb.init(project=self.project_name, name=self.exp_name, config=vars(self))
                self.wandb_run_id = self.run.id
                print(f"Created new wandb run: {self.wandb_run_id}")

            # Always save the current run_id for future resuming
            wandb_state = {"run_id": self.wandb_run_id}
            with open(wandb_state_path, "w") as f:
                json.dump(wandb_state, f)

    def _init_amp(self):
        self.amp_dtype = getattr(torch, self.amp_dtype)

    def _init_checkpointer(self):
        args = {
            "log_path": self.log_path_dir,
            "exp_name": self.exp_name,
            "save_checkpoint_path": self.save_checkpoint_path,
            "load_checkpoint_path": self.load_checkpoint_path,
            "dataset": self.dataset,
            "backbone_type": self.backbone_type,
            "resume": self.resume,
        }

        self.checkpointer = Checkpointer(**args)

    def _save_checkpoint(
        self,
        step,
        logs,
        new_checkpoint,
    ):
        # DeepSpeed handles model checkpointing
        if self.rank == 0:  # Only save on main process
            self.checkpointer.save(
                self.unwrap_model(self.backbone),
                self.optimizer.optimizer,
                self.lr_scheduler,
                self.ema,
                logs,
                step,
                new_checkpoint,
            )

    def _save_hparams(self):
        hparams_config_path = self.log_path / "hparams.json"
        hparams = dict(
            {
                "model_type": self.model_type,
                "backbone_type": self.backbone_type,
                "autoencoder_type": self.autoencoder_hparams["autoencoder_type"],
                "backbone_hparams": self.backbone_hparams,
                "cfg_scale": self.cfg_scale,
            }
        )
        with hparams_config_path.open("w") as f:
            json.dump(hparams, f)

    def _init_backbone(self):
        if self.is_main_process():
            print(self.autoencoder)

        if self.autoencoder_hparams["autoencoder_type"] == "llamagen":
            # LlamaGen VQ model has quantize.embedding.weight like standard quantized models
            num_embeddings = self.autoencoder.quantize.n_e
        else:
            num_embeddings = self.autoencoder.quantize.embedding.weight.shape[0]

        self.vocabulary_size = -1  # default value used for the methods that do not work with indices / tokens
        if self.model_type == "dfm":
            self.backbone_hparams["in_channels"] = 1
            self.vocabulary_size = num_embeddings + 1  # one token is for the [MSK]
        elif self.model_type == "purrception":
            self.vocabulary_size = num_embeddings

        # Initialize the backbone
        if self.backbone_type == "dit":
            self.backbone = DiT(
                model_type=self.model_type, vocabulary_size=self.vocabulary_size, **self.backbone_hparams
            )
        else:
            raise ValueError(f"Backbone {self.backbone_type} not implemented.")

        # Print information about the backbone
        if self.is_main_process():
            print(f"Number of parameters: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")
            print(f"Initialized {self.backbone_type} backbone.")

    def _init_dataloader(self):
        autoencoder_checkpoint_path = self.autoencoder_hparams["autoencoder_checkpoint_path"]

        # Determine vq_latent name from checkpoint path (imagenet only)
        if autoencoder_checkpoint_path.startswith("http://") or autoencoder_checkpoint_path.startswith("https://"):
            if "llamagen" in autoencoder_checkpoint_path.lower() or "vq_ds8_c2i" in autoencoder_checkpoint_path.lower():
                vq_latent = "vq-ds8-c2i"
            else:
                vq_latent = autoencoder_checkpoint_path.split("/")[-1].split(".")[0] or "vq-f8-n256"
        else:
            vq_latent = autoencoder_checkpoint_path.split("/")[-1].split(".")[0]
            if vq_latent == "":
                vq_latent = (
                    autoencoder_checkpoint_path.split("/")[-1]
                    if "/" in autoencoder_checkpoint_path
                    else autoencoder_checkpoint_path
                )
            if "llamagen" in autoencoder_checkpoint_path.lower() or "vq_ds8_c2i" in autoencoder_checkpoint_path.lower():
                vq_latent = "vq-ds8-c2i"

        if vq_latent == "vq-ds8-c2i":
            train_shards_path = f"./data/latents/imagenet/{vq_latent}/" + "shard_{000000..000421}.tar"
        elif vq_latent == "vq-f8":
            train_shards_path = f"./data/latents/imagenet/{vq_latent}/" + "shard_{000000..000211}.tar"
        else:
            raise ValueError(f"VQ latent {vq_latent} not implemented for imagenet.")

        self.dataloader = SimpleImageDataset(
            train_shards_path=train_shards_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_classes=self.backbone_hparams["num_classes"],
            data_fraction=self.data_fraction,
        )
        self.dataloader = self.dataloader.train_dataloader()

    def _init_optimizer(self):
        if self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.backbone.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay,
                eps=self.eps,
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not implemented.")

    def _init_lr_scheduler(self, last_epoch: int = -1):
        self.lr_scheduler = None

    def _prepare_autoencoder(self):
        if self.autoencoder_hparams["autoencoder_type"] == "stablediffusion":
            autoencoder_config_path = self.autoencoder_hparams["autoencoder_config_path"]
            autoencoder_checkpoint_path = self.autoencoder_hparams["autoencoder_checkpoint_path"]

            sys.path.insert(0, os.path.abspath("./latent_diffusion"))
            config = OmegaConf.load(autoencoder_config_path)
            autoencoder = instantiate_from_config(config.model)
            if self.is_main_process():
                print(f"Initialized VQ Model with configs from {autoencoder_config_path}.")
                print("The pretrained VQ Model has not been loaded (yet).")

            if autoencoder_checkpoint_path:
                pl_sd = torch.load(autoencoder_checkpoint_path, map_location="cpu", weights_only=False)
                sd = pl_sd["state_dict"]

                autoencoder.load_state_dict(sd, strict=False)
                if self.is_main_process():
                    print(f"Loaded pretrained VQ Model from {autoencoder_checkpoint_path}.")

                autoencoder.eval()
                autoencoder.requires_grad_(False)
        elif self.autoencoder_hparams["autoencoder_type"] == "llamagen":
            if not LLAMAGEN_AVAILABLE:
                raise ImportError("LlamaGen is not available. Cannot load llamagen autoencoder type.")

            autoencoder_checkpoint_path = self.autoencoder_hparams["autoencoder_checkpoint_path"]

            # Download checkpoint if it's a URL
            if autoencoder_checkpoint_path.startswith("http"):
                import urllib.request

                local_path = "./vq_ds8_c2i.pt"
                if not os.path.exists(local_path):
                    if self.is_main_process():
                        print(f"Downloading checkpoint from {autoencoder_checkpoint_path}...")
                    urllib.request.urlretrieve(autoencoder_checkpoint_path, local_path)
                autoencoder_checkpoint_path = local_path

            # Create model - VQ-8 for ds8 (downsample 8x)
            autoencoder = VQ_models["VQ-8"](codebook_size=16384, codebook_embed_dim=8)
            autoencoder.eval()

            # Load checkpoint
            checkpoint = torch.load(autoencoder_checkpoint_path, map_location="cpu", weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    # DDP format (most common for LlamaGen)
                    checkpoint = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    # PyTorch Lightning format
                    checkpoint = checkpoint["state_dict"]
                elif "ema" in checkpoint:
                    # EMA format
                    checkpoint = checkpoint["ema"]

            autoencoder.load_state_dict(checkpoint, strict=True)
            autoencoder.requires_grad_(False)

            if self.is_main_process():
                print(f"Initialized LlamaGen VQ model with checkpoint from {autoencoder_checkpoint_path}")
                print(f"  Codebook size: {autoencoder.quantize.n_e}, Embed dim: {autoencoder.quantize.e_dim}")
        else:
            raise NotImplementedError(
                f"Autoencoder type {self.autoencoder_hparams['autoencoder_type']} not implemented."
            )

        self.autoencoder = autoencoder.to(self.device)

    def _quantize_tensor(self, z, codebook):
        B, C, H, W = z.shape

        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, C)  # C - hidden dimension = 4

        dist = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(codebook, "n d -> d n"))
        )

        z_indices = torch.argmin(dist, dim=1)  # from the normalized codebook
        z_indices = rearrange(
            z_indices,
            "(b h w) -> b h w",
            b=self.batch_size,
            h=self.backbone_hparams["input_size"],
            w=self.backbone_hparams["input_size"],
        ).detach()

        return z_indices

    # @torch.compile
    def _train_step(self, z1, y, step, z1_indices, sampling_eps=1e-3):
        batch_size = z1.shape[0]

        # 1. Initialize the noise: a) Gaussian noise or b) [MSK] tokens (id = 0) for DFM
        if self.model_type in ["purrception", "cfm", "cfm-endpoint"]:
            z0 = torch.randn_like(z1)
        elif self.model_type == "dfm":
            z0 = torch.zeros_like(z1)
            # Ensure correct masking index (last one)
            mask_token_index = self.vocabulary_size - 1
            z0[:, mask_token_index, :, :] = 1.0  # Assign 1.0 to the mask token channel
        else:
            raise ValueError(f"Unknown model type for noise init: {self.model_type}")

        # 2. Sample t from Unif(0, 1) - ensure it has the same dtype as the model
        backbone_dtype = next(self.backbone.parameters()).dtype
        t = (1 - sampling_eps) * torch.rand(batch_size, device=self.device, dtype=backbone_dtype) + sampling_eps
        t = t.view(batch_size, 1, 1, 1)

        # 3. Compute zt as: a) Sampling for DFM or b) linear interpolation between z0 and z1 otherwise
        zt = t * z1 + (1 - t) * z0

        if self.model_type == "dfm":
            zt = sample_p(zt)  # [B, H, W] - tensor of indices

        t = rearrange(t, "b c h w -> (b c h w)")

        # 4. Get the predictions
        out = self.backbone(zt, t, y, self.cfg_scale)

        # 5. Compute the loss
        if self.model_type == "dfm":
            z_loss = 0.0
            if self.use_log_z:
                # Use log-sum-exp trick to avoid numerical issues
                log_z = torch.logsumexp(out, dim=1)
                z_loss = self.lambda_z * torch.mean(log_z.pow(2))
            loss = F.cross_entropy(out, z1_indices.long()) + z_loss
        elif self.model_type == "cfm":
            loss = F.mse_loss(out, z1 - z0)  # learns the velocity field
        elif self.model_type == "cfm-endpoint":
            loss = F.mse_loss(out, z1)  # learns the endpoint
        elif self.model_type == "purrception":
            z_loss = 0.0
            if self.use_log_z:
                # Use log-sum-exp trick to avoid numerical issues
                log_z = torch.logsumexp(out, dim=1)
                z_loss = self.lambda_z * torch.mean(log_z.pow(2))
            loss = F.cross_entropy(out, z1_indices.long()) + z_loss

        return loss

    def _init_deepspeed(self):
        """Initialize DeepSpeed engine"""
        # Get local rank from environment
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))

        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Only initialize process group if not already initialized
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        if self.is_main_process():
            print(f"Training on {self.device} with {self.world_size} processes.")
            print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
            effective_batch_size = self.batch_size * self.gradient_accumulation_steps * self.world_size
            print(f"Effective Batch Size: {effective_batch_size}.")

    def _prepare_models(self):
        """Prepare models for DeepSpeed"""
        # Move models to device
        self.backbone.to(self.device)
        self.autoencoder.to(self.device)

        # Initialize your optimizer first
        self._init_optimizer()

        # Load DeepSpeed config from JSON
        with open(self.ds_config_path, "r") as f:
            ds_config = json.load(f)

        # Set actual batch size values
        ds_config["train_micro_batch_size_per_gpu"] = self.batch_size
        ds_config["train_batch_size"] = self.batch_size * self.world_size * self.gradient_accumulation_steps
        ds_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps

        ds_config["gradient_clipping"] = self.max_grad_norm

        # Resume training state if resume argument set to true
        self.train_state = dict()
        if self.resume:
            self.checkpointer.maybe_load_state(
                backbone=self.backbone,
                optimizer=None,  # Do not load optimizer here
                lr_scheduler=self.lr_scheduler,
                train_state=self.train_state,
                ema=self.ema,
            )
        else:
            self.checkpointer.maybe_load_state(
                backbone=self.backbone,
            )

        # Initialize DeepSpeed engine with your optimizer
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.backbone,
            config=ds_config,
            optimizer=self.optimizer,  # Pass your existing optimizer
            lr_scheduler=self.lr_scheduler,
        )

        # Resume optimizer state if resuming training
        if self.resume:
            self.checkpointer.maybe_load_state(
                optimizer=optimizer.optimizer,
            )

        self.backbone = model_engine
        self.optimizer = optimizer

        if self.is_main_process():
            print(self.optimizer.optimizer.state_dict())

        # Don't wrap autoencoder in DDP
        self.autoencoder.to(self.device)

        if self.use_ema:
            self.ema.to(self.device)

    def is_main_process(self):
        return self.rank == 0

    def unwrap_model(self, model):
        if hasattr(model, "module"):
            return model.module
        return model

    def gather(self, tensor):
        if self.world_size > 1:
            # Handle scalar tensors (zero-dimensional)
            if tensor.dim() == 0:
                gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, tensor)
                return torch.stack(gathered)  # Use stack instead of cat for scalars
            else:
                gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gathered, tensor)
                return torch.cat(gathered, dim=0)
        return tensor

    def train(self):
        train_dataloader_iter = iter(self.dataloader)
        self.backbone.train()

        last_step = self.train_state.get("step", 0)
        logs = self.train_state.get("logs", dict())
        train_logs = logs.get("train_logs", [])

        if self.is_main_process():
            print(f"Starting training from step {last_step + 1}")

        # Track loss over accumulation steps
        accumulated_loss = 0.0
        accumulation_count = 0

        for step in trange(
            last_step + 1,
            self.num_iterations + 1,
            initial=last_step + 1,
            total=self.num_iterations,
            desc="Training",
            disable=not self.is_main_process(),  # Only show progress bar on main process
        ):
            batch = next(train_dataloader_iter)

            # Get the image and label (imagenet)
            z1, y = batch["latent"].to(self.device), batch["cls_id"].to(self.device)

            with torch.no_grad():
                if self.autoencoder_hparams["autoencoder_type"] in ["stablediffusion", "llamagen"]:
                    # Encode the input in latent space
                    # z1 has the shape [bs, in_chans, H, W] and it is already the quantized latent representation
                    # z1, _, (_, _, indices) = self.autoencoder.encode(x1)
                    z1, _, (_, _, indices) = self.autoencoder.quantize(z1)

                    indices = rearrange(
                        indices,
                        "(b h w) -> b h w",
                        b=self.batch_size,
                        h=self.backbone_hparams["input_size"],
                        w=self.backbone_hparams["input_size"],
                    ).detach()

                    if self.model_type == "dfm":
                        # One-hot encode the vector of indices
                        z1 = F.one_hot(indices.long(), self.vocabulary_size)  # shape [bs, H, W, vocabulary_size]
                        z1 = rearrange(z1, "b h w c -> b c h w")

            # Set dtype depending on the AMP
            z1 = z1.to(dtype=self.amp_dtype).detach()

            # Forward pass
            loss = self._train_step(z1, y, step, indices)

            # DeepSpeed backward and step - this handles accumulation automatically
            self.backbone.backward(loss)

            # Check if this completes a gradient accumulation cycle
            # DeepSpeed increments its internal micro_step counter
            is_accumulation_boundary = step % self.gradient_accumulation_steps == 0

            self.backbone.step()

            # Accumulate loss for logging (gather across devices)
            step_loss = self.gather(loss).mean().item()
            accumulated_loss += step_loss
            accumulation_count += 1

            # Log and update EMA only at accumulation boundaries
            if is_accumulation_boundary and self.is_main_process():
                # Average loss over the accumulation steps
                avg_loss = accumulated_loss / accumulation_count
                train_logs.append(avg_loss)

                print(f"Step {step}: {avg_loss}")

                # Prepare logging dictionary
                log_dict = {
                    "train/loss": avg_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "train/grad_norm": self.backbone.get_global_grad_norm(),
                    "train/max_grad_norm": self.max_grad_norm,
                }

                self.run.log(log_dict, step=step)

                # Reset accumulation tracking
                accumulated_loss = 0.0
                accumulation_count = 0

            # Update EMA after each gradient accumulation cycle
            if is_accumulation_boundary and self.use_ema:
                self.ema.update()

            # Evaluation and checkpointing (only on main process)
            if self.is_main_process():
                # Use EMA weights for sampling
                if step % self.eval_every_n_steps == 0 or step == self.num_iterations:
                    context = self.ema.average_parameters() if self.use_ema else contextlib.nullcontext()
                    with context:
                        # Get some samples (qualitative analysis during training)
                        samples = self.sample(
                            self.num_samples,
                            self.autoencoder,
                            self.backbone,
                            self.backbone_hparams,
                            self.ode_method,
                            self.ode_steps,
                            self.cfg_scale,
                            y=None,
                            temperature=1.0,
                        )
                        img_grid = torchvision.utils.make_grid(samples, nrow=samples.size(0))

                        self.run.log({"Samples": wandb.Image(img_grid)}, step=step)

                # Checkpointing
                if (
                    self.save_every_n_steps is not None and step % self.save_every_n_steps == 0
                ) or step == self.num_iterations:
                    logs = {"train_logs": train_logs}
                    self._save_checkpoint(step, logs, False)

                if self.save_new_every_n_steps is not None and step % self.save_new_every_n_steps == 0:
                    logs = {"train_logs": train_logs}
                    self._save_checkpoint(step, logs, True)
