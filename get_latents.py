import argparse
import os
import sys

import torch
import torch.distributed as dist
import webdataset as wds
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from dataset.imagenet import ImageNetDataset
from latent_diffusion.ldm.util import instantiate_from_config

# Add LlamaGen to path for vq-ds8-c2i support
sys.path.insert(0, os.path.abspath("./LlamaGen"))
try:
    from tokenizer.tokenizer_image.vq_model import VQ_models

    LLAMAGEN_AVAILABLE = True
except ImportError:
    LLAMAGEN_AVAILABLE = False
    print("Warning: LlamaGen not available. vq-ds8-c2i will not work.")


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # Use LOCAL_RANK for device assignment (GPU on this node)
        # LOCAL_RANK is set by torchrun/slurm
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    print(f"Global rank: {rank}, Local rank: {local_rank}, World size: {world_size}")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def encode_latents(autoencoder, images, mini_bs=32):
    # images: torch.Tensor, shape (B, 3, H, W)
    # Range: [-1, 1] for standard VQ and LlamaGen, [0, 1] for MaskGIT
    # Verify reasonable range (allow both [0, 1] and [-1, 1])
    assert (
        images.min() >= -1 and images.max() <= 1
    ), f"Images out of expected range: min={images.min()}, max={images.max()}"

    # Check if it's LlamaGen VQ model
    # LlamaGen VQModel is from tokenizer.tokenizer_image.vq_model module
    # It has a 'config' attribute (ModelArgs) and quantize with n_e
    module_name = getattr(type(autoencoder), "__module__", "")
    is_llamagen = (
        hasattr(autoencoder, "config")
        and hasattr(autoencoder, "quantize")
        and hasattr(autoencoder.quantize, "n_e")
        and "tokenizer_image.vq_model" in module_name
    )

    latents = []
    for i in range(0, len(images), mini_bs):
        _img = images[i : i + mini_bs]

        if is_llamagen:
            # LlamaGen VQ model: encode returns (quant, emb_loss, info)
            # where quant is the quantized latent tensor (B, codebook_embed_dim, H//8, W//8)
            # info = (perplexity, min_encodings, min_encoding_indices)
            # We extract the quantized latent embeddings (continuous codebook embeddings)
            _lat, _, _ = autoencoder.encode(_img)
            latents.append(_lat)
        else:
            # Standard VQ autoencoder (from latent_diffusion)
            # encode returns (quant, emb_loss, info) where info = (perplexity, min_encodings, min_encoding_indices)
            _lat, _, (_, _, indices) = autoencoder.encode(_img)
            latents.append(_lat)

    return torch.cat(latents, dim=0)


def prepare_autoencoder(
    autoencoder_checkpoint_path: str,
    autoencoder_config_path: str,
    device="cuda",
):
    sys.path.insert(0, os.path.abspath("./latent_diffusion"))
    config = OmegaConf.load(autoencoder_config_path)
    autoencoder = instantiate_from_config(config.model)

    print(f"Initialized VQ Model with configs from {autoencoder_config_path}.")
    print("The pretrained VQ Model has not been loaded (yet).")

    if autoencoder_checkpoint_path:
        pl_sd = torch.load(autoencoder_checkpoint_path, map_location="cpu", weights_only=False)
        sd = pl_sd["state_dict"]

        autoencoder.load_state_dict(sd, strict=False)

        print(f"Loaded pretrained VQ Model from {autoencoder_checkpoint_path}.")

        autoencoder.eval()
        autoencoder.requires_grad_(False)

    return autoencoder.to(device)


def prepare_llamagen_autoencoder(
    checkpoint_path: str = "./vq_ds8_c2i.pt",
    vq_model: str = "VQ-8",
    codebook_size: int = 16384,
    codebook_embed_dim: int = 8,
    device="cuda",
):
    """
    Load LlamaGen VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path or URL to the checkpoint
        vq_model: VQ model type ("VQ-8" for ds8, "VQ-16" for ds16)
        codebook_size: Codebook size (default: 16384)
        codebook_embed_dim: Codebook embedding dimension (default: 8)
        device: Device to load model on
    """
    if not LLAMAGEN_AVAILABLE:
        raise ImportError("LlamaGen is not available. Cannot load vq-ds8-c2i model.")

    # Download checkpoint if it's a URL
    if checkpoint_path.startswith("http"):
        import urllib.request

        local_path = "./vq_ds8_c2i.pt"
        if not os.path.exists(local_path):
            print(f"Downloading checkpoint from {checkpoint_path}...")
            urllib.request.urlretrieve(checkpoint_path, local_path)
        checkpoint_path = local_path

    # Create model
    autoencoder = VQ_models[vq_model](codebook_size=codebook_size, codebook_embed_dim=codebook_embed_dim)
    autoencoder.to(device)
    autoencoder.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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

    print(f"Loaded LlamaGen VQ Model from {checkpoint_path}")
    print(f"  Model: {vq_model}, Codebook size: {codebook_size}, Embed dim: {codebook_embed_dim}")

    return autoencoder


def shard_imagenet(batch_size: int, output_dir: str, vq_latent: str, num_workers: int = 8, max_tar_size=0.1):
    # [-1, 1] normalization for vq-f8 and vq-ds8-c2i
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset = ImageNetDataset(
        split="train",
        transform=transforms.Compose(transform_list),
    )

    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(local_rank)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
    )

    autoencoder_checkpoint_path = None
    autoencoder_config_path = None

    if vq_latent == "vq-f8":
        autoencoder_checkpoint_path = "autoencoders/vq-f8.ckpt"
        autoencoder_config_path = "latent_diffusion/models/first_stage_models/vq-f8/config.yaml"
    elif vq_latent == "vq-ds8-c2i":
        # LlamaGen vq_ds8_c2i model
        autoencoder_checkpoint_path = "https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds8_c2i.pt"
        autoencoder_config_path = None
    else:
        raise ValueError(f"vq_latent {vq_latent} is not valid!")

    print(f"autoencoder_checkpoint_path: {autoencoder_checkpoint_path}")
    print(f"autoencoder_config_path: {autoencoder_config_path}")

    if vq_latent == "vq-ds8-c2i":
        autoencoder = prepare_llamagen_autoencoder(
            checkpoint_path=autoencoder_checkpoint_path,
            vq_model="VQ-8",
            codebook_size=16384,
            codebook_embed_dim=8,
            device=device,
        )
    elif vq_latent == "vq-f8":
        autoencoder = prepare_autoencoder(
            autoencoder_checkpoint_path=autoencoder_checkpoint_path,
            autoencoder_config_path=autoencoder_config_path,
            device=device,
        )
    else:
        raise ValueError(f"vq_latent {vq_latent} is not valid!")
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        writer = wds.ShardWriter(
            os.path.join(output_dir, "shard_%06d.tar"),
            maxcount=1e6,
            maxsize=1e9 * max_tar_size,
        )
        sample_idx = 0

    try:
        for images, labels in tqdm(dataloader) if rank == 0 else dataloader:
            images = images.to(device)
            latents = encode_latents(autoencoder, images)
            latents_np = latents.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Gather all latents, labels, and images to rank 0
            gathered_latents = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            if world_size > 1:
                dist.all_gather_object(gathered_latents, latents_np)
                dist.all_gather_object(gathered_labels, labels_np)
            elif world_size == 1:
                gathered_latents = [latents_np]
                gathered_labels = [labels_np]

            if rank == 0:
                for gpu_latents, gpu_labels in tqdm(zip(gathered_latents, gathered_labels)):
                    for latent, label in zip(gpu_latents, gpu_labels):
                        sample = {
                            "__key__": f"{sample_idx:08d}",
                            "latent.npy": latent.astype("float32"),
                            "cls_id.cls": int(label),
                        }

                        writer.write(sample)
                        sample_idx += 1
    finally:
        if rank == 0:
            writer.close()
        cleanup_distributed()


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode latents")

    parser.add_argument(
        "--vq_latent",
        type=str,
        choices=["vq-f8", "vq-ds8-c2i"],
        required=True,
        help="Type of VQ latent space",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing the dataset",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Num workers for processing the dataset",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    vq_latent = args.vq_latent
    num_workers = args.num_workers

    output_dir = f"./data/latents/imagenet/{vq_latent}"
    shard_imagenet(batch_size=batch_size, num_workers=num_workers, vq_latent=vq_latent, output_dir=output_dir)
