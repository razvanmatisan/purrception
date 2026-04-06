import argparse

from trainer import Trainer
from utils.train_utils import get_exp_name, load_config


# Line command arguments for DiT
def add_backbone_args(backbone_args: argparse._ArgumentGroup):
    """
    Arguments that will be passed to initialize the backbone
    """
    backbone_args.add_argument("--input_size", type=int, default=32, help="Image size")
    backbone_args.add_argument("--patch_size", type=int, default=2, help="Patch size")
    backbone_args.add_argument("--in_channels", type=int, default=4, help="Number of input channels")

    backbone_args.add_argument("--hidden_size", type=int, default=768, help="Embedding dimension")
    backbone_args.add_argument("--depth", type=int, default=12, help="Number of transformer blocks from dit")
    backbone_args.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    backbone_args.add_argument("--mlp_ratio", type=int, default=4, help="MLP ratio")
    backbone_args.add_argument(
        "--num_classes",
        type=int,
        default=-1,
        help="Number of classes for conditioning argument (default is -1 for unconditional generation).",
    )


# Line command arguments for VQ latent space
def add_autoencoder_args(autoencoder_args: argparse._ArgumentGroup):
    """
    Arguments that will be passed to initialize the VQ model / AutoencoderKL
    """
    autoencoder_args.add_argument("--autoencoder_type", type=str, choices=["stablediffusion", "llamagen"])
    autoencoder_args.add_argument("--autoencoder_checkpoint_path", type=str, help="Path to the pretrained autoencoder")
    autoencoder_args.add_argument(
        "--autoencoder_config_path",
        type=str,
        default="latent_diffusion/models/first_stage_models/vq-f8/config.yaml",
        help="Path to the config file of the autoencoder.",
    )


# Line command arguments for training hyperparameters
def add_trainer_args(trainer_args: argparse._ArgumentGroup):
    """
    Arguments for training
    """
    trainer_args.add_argument("--num_iterations", type=int, help="Number of total training iterations")
    trainer_args.add_argument("--batch_size", type=int, help="Batch size (for each GPU)")
    trainer_args.add_argument("--use_amp", action="store_true", default=False, help="Use AMP")
    trainer_args.add_argument(
        "--amp_dtype", type=str, default="bfloat16", help="AMP data type", choices=["bfloat16", "float32"]
    )
    trainer_args.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    trainer_args.add_argument("--eval_every_n_steps", type=int, default=None, help="Eval every n steps")
    trainer_args.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps"
    )

    trainer_args.add_argument(
        "--temperature_softmax", action="store_true", default=None, help="Whether to apply softmax with temperature"
    )

    # Exponential Moving Average
    trainer_args.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether to use exponential moving average for model parameters",
    )

    trainer_args.add_argument(
        "--decay_ema",
        type=float,
        default=0.9999,
        help="Decay rate for the exponential moving average of the model parameters",
    )

    # Log_Z
    trainer_args.add_argument(
        "--use_log_z",
        action="store_true",
        default=False,
        help="Whether to use the log_z loss term",
    )

    trainer_args.add_argument(
        "--lambda_z",
        type=float,
        default=1e-5,
        help="Weight for the log_z loss (if use_log_z is True, otherwise ignored)",
    )

    # Optimizer
    trainer_args.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw"],
        help="Optimizer name",
    )
    trainer_args.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    trainer_args.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    trainer_args.add_argument("--beta1", type=float, default=0.9, help="Beta_1")
    trainer_args.add_argument("--beta2", type=float, default=0.999, help="Beta_2")
    trainer_args.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW optimizer")

    trainer_args.add_argument(
        "--num_warmup_steps",
        type=int,
        default=10000,
        help="Number of lr scheduler warmup steps",
    )

    trainer_args.add_argument("--num_workers", type=int, default=8, help="Number of workers for the dataloader")
    trainer_args.add_argument("--data_fraction", type=float, default=None, help="The percentage of the data we use.")

    # Logging
    trainer_args.add_argument("--log_path_dir", type=str, default="logs", help="Directory for logs")
    trainer_args.add_argument("--exp_name", type=str, default=None, help="Directory for experiment logs")

    # Checkpointing
    trainer_args.add_argument(
        "--project_name",
        type=str,
        default="variational-flow-matching-for-high-resolution-image-generation",
        help="The name of the project in wandb",
    )
    trainer_args.add_argument(
        "--save_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for saving the training state (log_path/exp_name/save_checkpoint_path)",
    )
    trainer_args.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="Frequency of saving the checkpoint",
    )
    trainer_args.add_argument(
        "--save_new_every_n_steps",
        type=int,
        default=None,
        help="Frequency of creating a new checkpoint (not overwriting the last checkpoint)",
    )

    # Resume training
    trainer_args.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="If true, resume from the last checkpoint from --log_path",
    )
    trainer_args.add_argument(
        "--load_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for loading the training state",
    )
    trainer_args.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="WandB project ID of the previous run",
    )

    # Classifier-free generation
    trainer_args.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free generation scale")


# Line command arguments for sampling hyperparameters (used for qualitative analysis during training)
def add_sampling_args(sampling_args: argparse._ArgumentGroup):
    sampling_args.add_argument("--ode_steps", type=int, default=100, help="Number of steps used to solve the ODE.")
    sampling_args.add_argument("--ode_method", type=str, default="euler", help="Method used to solve the ODE.")
    sampling_args.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate at once.")


# Add DeepSpeed argument parser
def add_deepspeed_args(parser):
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--ds_config", type=str, default="zero2.json", help="DeepSpeed config path")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser = add_deepspeed_args(parser)

    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model config. Overwrites command line arguments with arguments from the config file.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["purrception", "cfm", "dfm", "cfm-endpoint"],
        help="cfm means conditional flow matching and dfm means discrete flow matching.",
    )

    parser.add_argument(
        "--backbone_type",
        type=str,
        choices=["dit"],
        help="Backbone: DiT.",
    )

    args, _ = parser.parse_known_args()

    backbone_args = parser.add_argument_group("backbone")
    add_backbone_args(backbone_args)

    autoencoder_args = parser.add_argument_group("autoencoder")
    add_autoencoder_args(autoencoder_args)

    trainer_args = parser.add_argument_group("trainer")
    add_trainer_args(trainer_args)

    sampling_args = parser.add_argument_group("sampling")
    add_sampling_args(sampling_args)

    args = parser.parse_args()

    backbone_hparams = {a.dest: getattr(args, a.dest, None) for a in backbone_args._group_actions}
    autoencoder_hparams = {a.dest: getattr(args, a.dest, None) for a in autoencoder_args._group_actions}
    trainer_kwargs = {a.dest: getattr(args, a.dest, None) for a in trainer_args._group_actions}
    sampling_kwargs = {a.dest: getattr(args, a.dest, None) for a in sampling_args._group_actions}

    return args, backbone_hparams, autoencoder_hparams, trainer_kwargs, sampling_kwargs


if __name__ == "__main__":
    args, backbone_hparams, autoencoder_hparams, trainer_kwargs, sampling_kwargs = parse_args()

    if hasattr(args, "config_path") and args.config_path is not None:
        config = load_config(args.config_path)
        print(config)
        args.model_type = config["model_type"]
        args.backbone_type = config["backbone_type"]
        backbone_hparams.update(config["backbone_args"])
        autoencoder_hparams.update(config["autoencoder_args"])
        trainer_kwargs.update(config["trainer_args"])
        sampling_kwargs.update(config["sampler_args"])

    if args.exp_name is None:
        trainer_kwargs["exp_name"] = get_exp_name(args.model_type, args.backbone_type)

    print(args, backbone_hparams, autoencoder_hparams, trainer_kwargs)

    model = Trainer(
        model_type=args.model_type,
        backbone_type=args.backbone_type,
        backbone_hparams=backbone_hparams,
        autoencoder_hparams=autoencoder_hparams,
        ds_config_path=args.ds_config,  # Pass the config path
        **trainer_kwargs,
        **sampling_kwargs,
    )
    model.train()
