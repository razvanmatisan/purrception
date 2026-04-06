import argparse
import os
import sys
from pathlib import Path
from time import localtime, strftime

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import wandb
from latent_diffusion.ldm.util import instantiate_from_config
from models.dit import DiT
from utils.evaluation_utils import save_images_parallel
from utils.sample_utils import sample_cfm, sample_cfm_endpoint, sample_dfm, sample_purrception
from utils.train_utils import load_config, seed_everything

# Add LlamaGen to path for llamagen autoencoder support
sys.path.insert(0, os.path.abspath("./LlamaGen"))
try:
    from tokenizer.tokenizer_image.vq_model import VQ_models

    LLAMAGEN_AVAILABLE = True
except ImportError:
    LLAMAGEN_AVAILABLE = False


def compute_fid_torch_fidelity(samples_path, num_samples):
    """
    Compute FID and other metrics using torch-fidelity library (ImageNet validation).

    Args:
        samples_path: Path to the generated samples
        num_samples: Number of samples to use for FID computation

    Returns:
        metrics_dict: Dictionary containing all computed metrics
    """
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError("torch-fidelity is not installed. Please install it with: pip install torch-fidelity")

    reference_path = "data/imagenet_256_images_validation"
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=str(samples_path),
        input2=reference_path,
        cuda=torch.cuda.is_available(),
        fid=True,
        isc=True,
        kid=True,
        prc=True,
        verbose=False,
    )
    return metrics_dict


def get_args():
    parser = argparse.ArgumentParser(description="Sampling")

    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--y", type=int, default=None, help="Class id on which the samples are conditionally generated")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of total samples to generate")

    parser.add_argument("--seed", type=int, default=None, help="Seed")

    parser.add_argument(
        "--backbone_checkpoint_path",
        type=str,
        help="Path to the trained backbone (deprecated, use --base_path and --steps)",
    )
    parser.add_argument(
        "--base_path", type=str, help="Base path for checkpoints (e.g., 'logs/experiment/imagenet_dit')"
    )
    parser.add_argument(
        "--steps", nargs="+", type=str, help="List of checkpoint steps to evaluate (e.g., 100000 200000 300000)"
    )

    parser.add_argument(
        "--t_max_t_min",
        type=float,
        default=1.0,
        help="Temperature for sampling (constant).",
    )

    parser.add_argument(
        "--samples_path",
        type=str,
        default="samples",
        help="Name of the samples path",
    )

    parser.add_argument(
        "--compute_fid",
        action="store_true",
        default=False,
        help="Compute FID (and related metrics) with torch-fidelity vs data/imagenet_256_images_validation",
    )

    parser.add_argument(
        "--ode_method",
        type=str,
        default=None,
        help="ODE solver",
    )

    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="Number of ODE steps",
    )

    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for ODE solver",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-7,
        help="Relative tolerance for ODE solver",
    )

    parser.add_argument(
        "--use_distributed",
        action="store_true",
        default=False,
        help="Enable distributed (multi-GPU) sampling",
    )

    parser.add_argument(
        "--cleanup_images",
        action="store_true",
        default=False,
        help="Delete image directories after each checkpoint evaluation to save disk space",
    )

    parser.add_argument(
        "--no_barriers",
        action="store_true",
        default=False,
        help="Disable distributed barriers (for debugging timeout issues)",
    )

    parser.add_argument(
        "--iteration_barriers",
        action="store_true",
        default=False,
        help="Add barriers after every sampling iteration (helps with synchronization but may be slower)",
    )

    parser.add_argument(
        "--use_sde",
        action="store_true",
        default=False,
        help="Use stochastic sampling (SDE) instead of deterministic ODE for purrception",
    )

    parser.add_argument(
        "--sigma_schedule",
        type=str,
        choices=["constant", "linear", "exponential"],
        default="constant",
        help="Noise schedule for stochastic sampling",
    )

    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.0,
        help="Minimum noise level for stochastic sampling",
    )

    parser.add_argument(
        "--sigma_max",
        type=float,
        default=1.0,
        help="Maximum noise level for stochastic sampling",
    )

    return parser.parse_args()


def load_parameters(config_path):
    config = load_config(config_path)

    return (
        config["model_type"],
        config["backbone_type"],
        config["backbone_args"],
        config["autoencoder_args"],
        config["sampler_args"],
    )


def load_autoencoder(autoencoder_type, autoencoder_config_path, autoencoder_checkpoint_path):
    autoencoder = None

    if autoencoder_type == "stablediffusion":
        sys.path.insert(0, os.path.abspath("./latent_diffusion"))
        vq_config = OmegaConf.load(autoencoder_config_path)
        autoencoder = instantiate_from_config(vq_config.model)
        print(f"Initialized VQ Model with configs from {autoencoder_config_path}.")
        print("The pretrained VQ Model has not been loaded (yet).")

        pl_sd = torch.load(autoencoder_checkpoint_path, map_location="cpu", weights_only=False)
        sd = pl_sd["state_dict"]

        autoencoder.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained VQ Model from {autoencoder_checkpoint_path}.")

        autoencoder.eval()
        autoencoder.requires_grad_(False)
    elif autoencoder_type == "llamagen":
        if not LLAMAGEN_AVAILABLE:
            raise ImportError("LlamaGen is not available. Cannot load llamagen autoencoder type.")

        # Download checkpoint if it's a URL
        if autoencoder_checkpoint_path.startswith("http"):
            import urllib.request

            local_path = "./vq_ds8_c2i.pt"
            if not os.path.exists(local_path):
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

        print(f"Initialized LlamaGen VQ model with checkpoint from {autoencoder_checkpoint_path}")
        print(f"  Codebook size: {autoencoder.quantize.n_e}, Embed dim: {autoencoder.quantize.e_dim}")
    else:
        raise NotImplementedError(f"Autoencoder type {autoencoder_type} not implemented.")

    return autoencoder


def load_backbone(model_type, backbone_type, autoencoder, backbone_checkpoint_path, **backbone_args):
    # Initialize the backbone
    vocabulary_size = -1

    # Get number of embeddings based on autoencoder type
    if hasattr(autoencoder, "quantize") and hasattr(autoencoder.quantize, "n_e"):
        # LlamaGen VQ model
        num_embeddings = autoencoder.quantize.n_e
    elif hasattr(autoencoder, "n_embed"):
        # Standard quantized model
        num_embeddings = autoencoder.n_embed
    else:
        raise ValueError(f"Could not determine vocabulary size from autoencoder: {type(autoencoder)}")

    if model_type == "dfm":
        backbone_args["in_channels"] = 1
        vocabulary_size = num_embeddings + 1  # one token is for the [MSK]
    elif model_type == "purrception":
        vocabulary_size = num_embeddings
    elif model_type in ["cfm", "cfm-endpoint"]:
        # CFM models work with continuous embeddings, in_channels should be set in config
        # vocabulary_size remains -1 (default)
        pass

    if backbone_type == "dit":
        backbone = DiT(model_type=model_type, vocabulary_size=vocabulary_size, **backbone_args)
    else:
        raise ValueError(f"{backbone_type} is not a valid backbone.")

    # Load the model state
    model_state = torch.load(backbone_checkpoint_path, map_location="cpu", weights_only=False)

    ema = None
    if model_state.get("ema_state_dict") is not None:
        print("Loading backbone state from EMA...")

        # First load the backbone state (either from EMA or regular checkpoint)
        if "backbone_state_dict" in model_state:
            backbone.load_state_dict(model_state["backbone_state_dict"])

        # Create and load EMA
        ema = ExponentialMovingAverage(backbone.parameters(), decay=0.9999)
        ema.load_state_dict(model_state["ema_state_dict"])

        # Apply EMA weights to backbone for inference
        ema.copy_to(backbone.parameters())
        print("Applied EMA weights to backbone for inference.")

    elif "backbone_state_dict" in model_state:
        print("Loading backbone state from checkpoint...")
        backbone.load_state_dict(model_state["backbone_state_dict"])
    else:
        print("WARNING: Backbone state not found in checkpoint. The backbone will be initialized from scratch.")
        exit(1)

    return backbone


def get_sample_function(model_type, use_sde=False):
    sample_fct = None

    if model_type == "cfm":
        sample_fct = sample_cfm
    elif model_type == "cfm-endpoint":
        sample_fct = sample_cfm_endpoint
    elif model_type == "dfm":
        sample_fct = sample_dfm
    elif model_type == "purrception":
        sample_fct = sample_purrception
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return sample_fct


def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set the device for this process BEFORE initializing process group
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Initialize the process group with explicit device_id and 30-minute timeout
        import datetime

        timeout = datetime.timedelta(minutes=30)
        dist.init_process_group(backend="nccl", device_id=device, timeout=timeout)

        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": device,
            "is_main_process": rank == 0,
            "distributed": True,
        }
    else:
        # Single GPU mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "rank": 0,
            "world_size": 1,
            "local_rank": 0,
            "device": device,
            "is_main_process": True,
            "distributed": False,
        }


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Setup distributed training
    dist_info = setup_distributed()
    device = dist_info["device"]
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    is_main_process = dist_info["is_main_process"]
    distributed = dist_info["distributed"]

    if distributed:
        print(f"Process {rank}/{world_size} on device {device}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 1. Load line command arguments
    args = get_args()

    config_path = args.config_path
    y = args.y
    base_path = args.base_path
    steps = args.steps
    backbone_checkpoint_path = args.backbone_checkpoint_path
    samples_path = args.samples_path
    num_samples = args.num_samples
    compute_fid = args.compute_fid

    use_distributed = args.use_distributed
    cleanup_images = args.cleanup_images
    no_barriers = args.no_barriers
    iteration_barriers = args.iteration_barriers

    atol = args.atol
    rtol = args.rtol

    # Stochastic sampling parameters
    use_sde = args.use_sde
    sigma_schedule = args.sigma_schedule
    sigma_min = args.sigma_min
    sigma_max = args.sigma_max

    # Override distributed setting based on environment
    if distributed and not use_distributed:
        print("Warning: Distributed environment detected but --use_distributed not set. Enabling distributed mode.")
        use_distributed = True
    elif use_distributed and not distributed:
        print("Warning: --use_distributed set but no distributed environment detected. Using single GPU.")
        use_distributed = False

    t_max_t_min = args.t_max_t_min

    # Validate arguments
    if base_path is None and backbone_checkpoint_path is None:
        raise ValueError("Either --base_path or --backbone_checkpoint_path must be provided")

    if base_path is not None and steps is None:
        raise ValueError("--steps must be provided when using --base_path")

    checkpoint_paths = []
    # If using old single checkpoint mode
    if backbone_checkpoint_path is not None:
        steps = [None]  # Single iteration
        checkpoint_paths = [backbone_checkpoint_path]
    else:
        # Build checkpoint paths from base_path and steps
        checkpoint_paths = [f"{base_path}_step-{step}.pth" for step in steps if step != "last"]

    if "last" in steps:
        checkpoint_paths.append(f"{base_path}_last.pth")

    if is_main_process:
        print(f"Temperature (t_max_t_min): {t_max_t_min}")
        print(f"Steps to evaluate: {steps}")
        print(f" Number of checkpoints to evaluate: {len(checkpoint_paths)}")
        print(f" Base path: {base_path}")
        print(f"Atol: {atol}")
        print(f"Rtol: {rtol}")
        print(f"Stochastic sampling (SDE): {use_sde}")
        if use_sde:
            print(f"  Sigma schedule: {sigma_schedule}")
            print(f"  Sigma min: {sigma_min}")
            print(f"  Sigma max: {sigma_max}")
        print(f"Checkpoint paths: {checkpoint_paths}")
        print(f"Distributed: {use_distributed} (World size: {world_size})")
        print(f"Cleanup images after processing: {cleanup_images}")
        print(f"Iteration barriers enabled: {iteration_barriers}")
        print(f"All barriers disabled: {no_barriers}")
        print(f"Compute FID (torch-fidelity): {compute_fid}")

    # 2. Create directory where samples are saved
    base_samples_path = Path(samples_path)
    base_samples_path.mkdir(parents=True, exist_ok=True)
    print(f"The samples will be saved under the base path {base_samples_path}")

    # 3. Load parameters from config file
    model_type, backbone_type, backbone_args, autoencoder_args, sampler_args = load_parameters(config_path)
    num_classes = backbone_args["num_classes"]

    total_number_samples = sampler_args["total_number_samples"]
    batch_size_per_gpu = sampler_args["batch_size_per_gpu"]
    ode_steps = sampler_args["ode_steps"]
    ode_method = sampler_args["ode_method"]
    cfg_scale = sampler_args["cfg_scale"]

    if args.ode_steps is not None:
        ode_steps = args.ode_steps

    if args.ode_method is not None:
        ode_method = args.ode_method

    if args.cfg_scale is not None:
        cfg_scale = args.cfg_scale

    if args.seed is not None:
        seed = args.seed
        seed_everything(seed)

    timestamp = strftime("%Y-%m-%d_%H:%M:%S", localtime())
    if is_main_process:
        sde_suffix = f"-sde_{sigma_schedule}_{sigma_max}" if use_sde else ""
        wandb.init(
            project="sampling",
            name=f"sampling-{timestamp}-{base_path}-temperature_{t_max_t_min}-cfg_{cfg_scale}-steps_{'_'.join(steps)}{sde_suffix}",
        )

    # For distributed GPU, we set batch size per GPU
    batch_size = batch_size_per_gpu  # Each GPU processes this batch size

    # Override num_samples if provided via command line
    if num_samples is not None:
        total_number_samples = num_samples

    # Calculate per-GPU samples for distributed training
    if use_distributed:
        assert (
            total_number_samples % world_size == 0
        ), f"Total samples ({total_number_samples}) must be divisible by world size ({world_size})"
        samples_per_gpu = total_number_samples // world_size
        assert (
            samples_per_gpu % batch_size == 0
        ), f"Samples per GPU ({samples_per_gpu}) must be divisible by batch size ({batch_size})"
        num_iterations = samples_per_gpu // batch_size
    else:
        assert total_number_samples % batch_size == 0
        num_iterations = total_number_samples // batch_size

    if y is None and num_classes > 0:
        if use_distributed:
            # Each GPU handles a subset of classes (not all classes)
            assert (
                num_classes % world_size == 0
            ), f"Number of classes ({num_classes}) must be divisible by world size ({world_size})"
            classes_per_gpu = num_classes // world_size

            # All samples for this GPU are distributed among its assigned classes
            assert (
                samples_per_gpu % classes_per_gpu == 0
            ), f"Samples per GPU ({samples_per_gpu}) must be divisible by classes per GPU ({classes_per_gpu})"
            samples_per_class = samples_per_gpu // classes_per_gpu

            # Each batch contains samples from multiple classes
            assert (
                batch_size % samples_per_class == 0
            ), f"Batch size ({batch_size}) must be divisible by samples per class ({samples_per_class})"
            classes_per_batch = batch_size // samples_per_class
        else:
            assert total_number_samples % num_classes == 0
            assert (batch_size * num_classes) % total_number_samples == 0
            samples_per_class = total_number_samples // num_classes
            classes_per_batch = batch_size // samples_per_class

    # 4. Load autoencoder (VQ-GAN or AutoencoderKL) - only once
    autoencoder = load_autoencoder(**autoencoder_args)
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    # Compile for better performance (optional)
    autoencoder.decode = torch.compile(autoencoder.decode, mode="reduce-overhead")
    # Only compile decode_tokens if it exists (not available for LlamaGen)
    if hasattr(autoencoder, "decode_tokens"):
        autoencoder.decode_tokens = torch.compile(autoencoder.decode_tokens, mode="reduce-overhead")

    # 7. Get sampling function
    sample_fct = get_sample_function(model_type, use_sde=use_sde)

    if is_main_process:
        print(f"Number iterations: {num_iterations}")
        print(f"Number GPUs used: {world_size}")
        print(f"Total number samples: {total_number_samples}")
        if use_distributed:
            print(f"Samples per GPU: {samples_per_gpu}")
            if y is None and num_classes > 0:
                print(f"Classes per GPU: {classes_per_gpu}")
                print(f"Samples per class: {samples_per_class}")
        elif y is None and num_classes > 0:
            print(f"Samples per class: {samples_per_class}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"ODE method: {ode_method}")
        print(f"ODE steps: {ode_steps}")
        print(f"CFG scale: {cfg_scale}")
        print("--------------------------------")

    # 5. Loop through each checkpoint step
    for step, checkpoint_path in zip(steps, checkpoint_paths):
        step = int(step) if step != "last" else "last"
        if is_main_process:
            print(f"\n{'='*50}")
            print(f"Processing checkpoint step: {step}")
            print(f"Checkpoint path: {checkpoint_path}")
            print(f"{'='*50}")

        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            if is_main_process:
                print(f"WARNING: Checkpoint {checkpoint_path} does not exist. Skipping...")
            continue

        # Create step-specific samples directory
        step_samples_path = (
            base_samples_path / f"{t_max_t_min}" / f"{cfg_scale}" / f"{ode_method}" / f"{ode_steps}" / f"step_{step}"
        )
        # All GPUs save to the same directory to avoid file copying bottleneck
        step_samples_path.mkdir(parents=True, exist_ok=True)

        if is_main_process:
            print(f"Samples for step {step} will be saved at: {step_samples_path}")

        # 6. Load backbone for this checkpoint
        backbone = load_backbone(model_type, backbone_type, autoencoder, checkpoint_path, **backbone_args)
        backbone = backbone.to(device)
        backbone.eval()

        # Synchronize all processes before sampling
        if use_distributed and not no_barriers:
            dist.barrier()

        # 8. Sampling for this checkpoint
        with torch.no_grad():
            if y is None:
                for idx in tqdm(range(0, num_iterations), desc=f"Sampling step {step} (rank {rank})"):
                    if num_classes > 0:
                        # Distribute classes across GPUs
                        if use_distributed:
                            # Each GPU handles a different subset of classes
                            start_class_for_gpu = rank * classes_per_gpu
                            classes = torch.arange(
                                start_class_for_gpu + idx * classes_per_batch,
                                start_class_for_gpu + (idx + 1) * classes_per_batch,
                            )
                            y_batch = torch.repeat_interleave(classes, repeats=samples_per_class)
                        else:
                            # Single GPU mode
                            classes = torch.arange(idx * classes_per_batch, (idx + 1) * classes_per_batch)
                            y_batch = torch.repeat_interleave(classes, repeats=samples_per_class)

                        images = sample_fct(
                            batch_size,
                            autoencoder,
                            backbone,
                            backbone_args,
                            ode_method,
                            ode_steps,
                            cfg_scale,
                            y_batch,
                            device=device,
                            temperature=t_max_t_min,
                            atol=atol,
                            rtol=rtol,
                            sigma_schedule=sigma_schedule,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                        )

                        # Calculate correct indices for this GPU
                        if use_distributed:
                            # Each GPU handles its own set of classes completely
                            # All samples for each class are generated by one GPU
                            # So naming should start from 0 for each class
                            idx_start = 0
                            idx_end = samples_per_class
                        else:
                            idx_start = 0
                            idx_end = samples_per_class

                        save_images_parallel(
                            images=images,
                            labels=classes,
                            path=step_samples_path,
                            idx_start=idx_start,
                            idx_end=idx_end,
                        )

                        # Synchronize after each iteration to keep processes aligned
                        if use_distributed and not no_barriers and iteration_barriers:
                            try:
                                dist.barrier()
                                if idx % 10 == 0:  # Print progress every 10 iterations
                                    print(f"Rank {rank}: Completed iteration {idx+1}/{num_iterations}")
                            except Exception as e:
                                print(f"Rank {rank}: Barrier timeout at iteration {idx+1}: {e}")

                    else:
                        images = sample_fct(
                            batch_size,
                            autoencoder,
                            backbone,
                            backbone_args,
                            ode_method,
                            ode_steps,
                            cfg_scale,
                            None,  # No class conditioning
                            device=device,
                            temperature=t_max_t_min,
                            atol=atol,
                            rtol=rtol,
                            sigma_schedule=sigma_schedule,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                        )

                        # Calculate indices for distributed sampling
                        if use_distributed:
                            idx_start = rank * samples_per_gpu + idx * batch_size
                            idx_end = rank * samples_per_gpu + (idx + 1) * batch_size
                        else:
                            idx_start = idx * batch_size
                            idx_end = (idx + 1) * batch_size

                        save_images_parallel(
                            images=images, labels=None, path=step_samples_path, idx_start=idx_start, idx_end=idx_end
                        )

                        # Synchronize after each iteration to keep processes aligned
                        if use_distributed and not no_barriers and iteration_barriers:
                            try:
                                dist.barrier()
                                if idx % 10 == 0:  # Print progress every 10 iterations
                                    print(f"Rank {rank}: Completed iteration {idx+1}/{num_iterations}")
                            except Exception as e:
                                print(f"Rank {rank}: Barrier timeout at iteration {idx+1}: {e}")

            elif y is not None and num_classes > 0 and num_samples is not None:
                # For conditional sampling with specific class
                actual_num_samples = samples_per_gpu if use_distributed else num_samples
                y_tensor = torch.ones(actual_num_samples, dtype=torch.long).to(device) * y

                images = sample_fct(
                    actual_num_samples,
                    autoencoder,
                    backbone,
                    backbone_args,
                    ode_method,
                    ode_steps,
                    cfg_scale,
                    y_tensor,
                    device=device,
                    temperature=t_max_t_min,
                    atol=atol,
                    rtol=rtol,
                    sigma_schedule=sigma_schedule,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                )

                # Calculate start index for this GPU
                idx_start = rank * samples_per_gpu if use_distributed else 0
                idx_end = idx_start + actual_num_samples

                save_images_parallel(
                    images=images, labels=None, path=step_samples_path, idx_start=idx_start, idx_end=idx_end
                )
            else:
                raise NotImplementedError("Not implemented")

        # Synchronize all processes after sampling (essential for correctness)
        # Skip barrier if FID will be computed (to avoid timeout during long FID computation)
        if use_distributed and not no_barriers and not compute_fid:
            try:
                dist.barrier()
                print(f"Rank {rank}: All ranks finished sampling for step {step}")
            except Exception as e:
                print(f"Rank {rank}: Barrier timeout after sampling: {e}")

        # 9. Compute FID for this checkpoint (only on main process)
        # Note: Other processes will continue without waiting for FID completion
        if compute_fid and is_main_process:
            print(f"Computing FID for step {step} (ImageNet) with torch-fidelity...")

            try:
                metrics_dict = compute_fid_torch_fidelity(step_samples_path, total_number_samples)

                print(f"Step {step} - Metrics (torch-fidelity):")
                for metric_name, metric_value in metrics_dict.items():
                    print(f"  {metric_name}: {metric_value}")

                wandb_step = step if step != "last" else 5000000  # BUG! Temporary fix for logging

                wandb_metrics = {}
                for metric_name, metric_value in metrics_dict.items():
                    wandb_name = f"torch_fid_{metric_name}"
                    wandb_metrics[wandb_name] = metric_value
                    wandb_metrics[f"{wandb_name}_{total_number_samples}"] = metric_value

                wandb.log(wandb_metrics, step=wandb_step)

            except Exception as e:
                print(f"Error computing FID with torch-fidelity: {e}")
                print("Make sure torch-fidelity is installed: pip install torch-fidelity")

        # Cleanup (only main process to avoid race conditions)
        # Do this immediately after FID without waiting for other processes
        if cleanup_images and is_main_process:
            print(f"Cleaning up image directories for step {step}...")
            import shutil

            # Remove the step samples directory (same for both distributed and single GPU)
            if step_samples_path.exists():
                shutil.rmtree(step_samples_path)
                print(f"  Removed {step_samples_path}")

            print(f"Cleanup completed for step {step}")

        # Clean up backbone to save memory
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Light synchronization - just to keep processes roughly together
        # Don't make this blocking for the entire workflow
        if use_distributed:
            print(f"Rank {rank}: Finished processing checkpoint {step}")

    # Final synchronization only at the very end
    # Skip final barrier if FID was computed (to avoid timeout - FID computation on main process
    # can take a very long time and other ranks shouldn't wait for it)
    if use_distributed and not no_barriers and not compute_fid:
        print(f"Rank {rank}: All checkpoints completed, final synchronization...")
        try:
            dist.barrier()
            print(f"Rank {rank}: All ranks finished all checkpoints")
        except Exception as e:
            print(f"Rank {rank}: Final barrier timeout/error: {e}")
            # Continue anyway
    elif use_distributed and compute_fid:
        # If FID was computed, just print completion without barrier
        print(f"Rank {rank}: All checkpoints completed (FID computation may still be running on main process)")

    if is_main_process:
        print("All checkpoints processed!")
        wandb.finish()

    # Clean up distributed training
    cleanup_distributed()


if __name__ == "__main__":
    main()
