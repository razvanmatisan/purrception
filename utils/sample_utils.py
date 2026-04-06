from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torchdiffeq import odeint_adjoint as odeint  # Uses adjoint method


def _is_llamagen_tokenizer(autoencoder):
    """Check if autoencoder is a LlamaGen VQ tokenizer."""
    # LlamaGen VQ models have quantize attribute with n_e and e_dim
    return (
        hasattr(autoencoder, "quantize")
        and hasattr(autoencoder.quantize, "n_e")
        and hasattr(autoencoder.quantize, "e_dim")
    )


def _get_codebook(autoencoder, device, backbone_dtype):
    """Get codebook from autoencoder, handling standard and llamagen tokenizers."""
    if _is_llamagen_tokenizer(autoencoder):
        # LlamaGen VQ model: codebook is quantize.embedding.weight
        codebook = autoencoder.quantize.embedding.weight  # [codebook_size, embedding_dim]
        codebook = codebook.to(dtype=backbone_dtype).to(device)
    else:
        num_embeddings = autoencoder.quantize.embedding.weight.shape[0]
        codebook = torch.arange(0, num_embeddings, dtype=torch.long, device=device)
        codebook = autoencoder.quantize.embedding(codebook)
        codebook = codebook.to(dtype=backbone_dtype)
    return codebook


def _get_num_embeddings(autoencoder):
    """Get number of embeddings from autoencoder."""
    if _is_llamagen_tokenizer(autoencoder):
        return autoencoder.quantize.n_e
    else:
        return autoencoder.quantize.embedding.weight.shape[0]


def _decode_indices(autoencoder, indices):
    """Decode indices to images, handling standard and llamagen tokenizers."""
    if _is_llamagen_tokenizer(autoencoder):
        # LlamaGen VQ model: use decode_code method
        # indices should be in shape [B, H*W] or [B, H, W]
        B = indices.shape[0]
        if indices.dim() == 2:
            # Shape: [B, H*W]
            H_W = indices.shape[1]
            H = W = int(H_W**0.5)
            indices_reshaped = indices.reshape(B, H, W)
        else:
            # Shape: [B, H, W]
            indices_reshaped = indices
            H, W = indices_reshaped.shape[1], indices_reshaped.shape[2]

        # Get embedding dimension
        embed_dim = autoencoder.quantize.e_dim
        qzshape = [B, embed_dim, H, W]

        # Flatten indices for decode_code: [B, H, W] -> [B, H*W]
        indices_flat = indices_reshaped.reshape(B, -1)

        # Decode using decode_code
        images = autoencoder.decode_code(indices_flat, qzshape)
        # LlamaGen returns images in [-1, 1] range
        images = images.clamp(-1, 1)
        images = ((images + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
    else:
        # Standard autoencoder expects decode_tokens method
        if hasattr(autoencoder, "decode_tokens"):
            images = autoencoder.decode_tokens(indices)
            images = images.clamp(-1, 1)
            images = ((images + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
        else:
            # Fallback to decode if decode_tokens doesn't exist
            # This would require converting indices to embeddings first
            raise NotImplementedError("Standard autoencoder decode_tokens not available")
    return images


def _decode_embeddings(autoencoder, embeddings):
    """Decode embeddings to images, handling standard and llamagen tokenizers."""
    if _is_llamagen_tokenizer(autoencoder):
        # For LlamaGen, we can decode embeddings directly using decode method
        # Or convert to indices first and use decode_code
        # Let's use decode directly since embeddings are already quantized
        images = autoencoder.decode(embeddings)
        # LlamaGen returns images in [-1, 1] range
        images = images.clamp(-1, 1)
        images = ((images + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
    else:
        # Standard autoencoder decode
        images = autoencoder.decode(embeddings)
        images = images.clamp(-1, 1)
        images = ((images + 1) * 0.5 * 255.0).to(dtype=torch.uint8)
    return images


def _pad_like_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *(1 for _ in range(y.ndim - x.ndim)))


def argmax_p(pt, xt, mask_token_id):
    """
    pt: (B, K, T)
    xt: (B, T)
    """

    pt[:, mask_token_id] = 0  # make mask_token_id never be the max
    max_xt = pt.argmax(dim=1)
    is_mask = xt == mask_token_id
    xt[is_mask] = max_xt[is_mask]
    _ratio = (is_mask.sum() / is_mask.numel()).item()
    print(f"finish argmax_p, max_last ratio: {_ratio}")
    return xt


@torch.compile
def sample_p(pt):
    B, C, H, W = pt.shape
    pt = rearrange(pt, "b c h w -> (b h w) c")
    zt = torch.multinomial(pt, 1)

    return zt.reshape(B, H, W)


def sample_cfm(
    num_samples,
    autoencoder,
    backbone,
    backbone_config,
    ode_method,
    ode_steps,
    cfg_scale,
    y,
    device="cuda:0",
    **kargs,
):
    in_channels = backbone_config["in_channels"]
    input_size = backbone_config["input_size"]
    num_classes = backbone_config["num_classes"]

    backbone_dtype = next(backbone.parameters()).dtype

    if num_classes > 0 and y is None:
        y = torch.randint(0, num_classes, (num_samples,))

    x0 = torch.randn(num_samples, in_channels, input_size, input_size, dtype=backbone_dtype).to(device)
    t = torch.linspace(0, 1, ode_steps, dtype=backbone_dtype).to(device)

    if num_classes > 0 and y is not None:
        y = y.to(device)

    x0 = rearrange(x0, "b c h w -> b (c h w)")

    ode_func = partial(
        _ode_function_cfm,
        y=y,
        backbone=backbone,
        num_samples=num_samples,
        input_size=input_size,
        cfg_scale=cfg_scale,
    )

    with torch.no_grad():
        trajectory = odeint(func=ode_func, y0=x0, t=t, method=ode_method, adjoint_params=())

        x1 = trajectory[-1]
        x1 = rearrange(x1, "b (c h w) -> b c h w", b=num_samples, h=input_size, w=input_size)

        if _is_llamagen_tokenizer(autoencoder):
            original_dtype = autoencoder.quantize.embedding.weight.dtype
            autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                dtype=backbone_dtype
            )
            try:
                x1_q, _, _ = autoencoder.quantize(x1)
            finally:
                autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                    dtype=original_dtype
                )
        else:
            original_dtype = autoencoder.quantize.embedding.weight.dtype
            autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                dtype=backbone_dtype
            )
            try:
                x1_q, _, _ = autoencoder.quantize(x1)
            finally:
                autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                    dtype=original_dtype
                )

        x1_q = x1_q.to(dtype=torch.float32)
        images = _decode_embeddings(autoencoder, x1_q)

        return images


def _ode_function_cfm(t, x, backbone, y, num_samples, input_size, cfg_scale=1.0):
    x = rearrange(x, "b (c h w) -> b c h w", b=num_samples, h=input_size, w=input_size)
    t = t.to(dtype=x.dtype)
    t_batch = t.expand(num_samples)
    mu = backbone(x, t_batch, y, cfg_scale)
    mu = rearrange(mu, "b c h w -> b (c h w)")
    return mu


def _ode_function_cfm_endpoint(t, x, backbone, y, num_samples, input_size, cfg_scale=1.0, eps=1e-6):
    """ODE function for CFM with endpoint prediction: (mu - x) / (1 - t + eps)"""
    x = rearrange(x, "b (c h w) -> b c h w", b=num_samples, h=input_size, w=input_size)
    t = t.to(dtype=x.dtype)
    t_batch = t.expand(num_samples)
    mu = backbone(x, t_batch, y, cfg_scale)
    x = rearrange(x, "b c h w -> b (c h w)")
    mu = rearrange(mu, "b c h w -> b (c h w)")
    return (mu - x) / (1 - t + eps)


def sample_cfm_endpoint(
    num_samples,
    autoencoder,
    backbone,
    backbone_config,
    ode_method,
    ode_steps,
    cfg_scale,
    y,
    device="cuda:0",
    **kargs,
):
    """
    Sample from CFM model that predicts the endpoint instead of velocity.
    Uses ODE: dx/dt = (mu - x) / (1 - t + eps) where mu is the predicted endpoint.
    """
    in_channels = backbone_config["in_channels"]
    input_size = backbone_config["input_size"]
    num_classes = backbone_config["num_classes"]

    backbone_dtype = next(backbone.parameters()).dtype

    if num_classes > 0 and y is None:
        y = torch.randint(0, num_classes, (num_samples,))

    x0 = torch.randn(num_samples, in_channels, input_size, input_size, dtype=backbone_dtype).to(device)
    t = torch.linspace(0, 1, ode_steps, dtype=backbone_dtype).to(device)

    if num_classes > 0 and y is not None:
        y = y.to(device)

    x0 = rearrange(x0, "b c h w -> b (c h w)")

    ode_func = partial(
        _ode_function_cfm_endpoint,
        y=y,
        backbone=backbone,
        num_samples=num_samples,
        input_size=input_size,
        cfg_scale=cfg_scale,
    )

    with torch.no_grad():
        trajectory = odeint(func=ode_func, y0=x0, t=t, method=ode_method, adjoint_params=())

        x1 = trajectory[-1]
        x1 = rearrange(x1, "b (c h w) -> b c h w", b=num_samples, h=input_size, w=input_size)

        if _is_llamagen_tokenizer(autoencoder):
            original_dtype = autoencoder.quantize.embedding.weight.dtype
            autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                dtype=backbone_dtype
            )
            try:
                x1_q, _, _ = autoencoder.quantize(x1)
            finally:
                autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                    dtype=original_dtype
                )
        else:
            original_dtype = autoencoder.quantize.embedding.weight.dtype
            autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                dtype=backbone_dtype
            )
            try:
                x1_q, _, _ = autoencoder.quantize(x1)
            finally:
                autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                    dtype=original_dtype
                )

        x1_q = x1_q.to(dtype=torch.float32)
        images = _decode_embeddings(autoencoder, x1_q)

        return images


def sample_dfm(
    num_samples,
    autoencoder,
    backbone,
    backbone_config,
    ode_method,
    euler_steps,
    cfg_scale,
    y,
    t_min=1e-4,
    device="cuda:0",
    **kargs,
):
    num_embeddings = _get_num_embeddings(autoencoder)
    vocabulary_size = num_embeddings + 1
    input_size = backbone_config["input_size"]
    num_classes = backbone_config["num_classes"]

    backbone_dtype = next(backbone.parameters()).dtype

    if num_classes > 0 and y is None:
        y = torch.randint(0, num_classes, (num_samples,)).to(device)

    # Get a tensor filled with only the [MSK] token
    x0 = (torch.ones(num_samples, input_size, input_size) * vocabulary_size - 1).to(device=device, dtype=torch.long)
    dirac_x0 = F.one_hot(x0, vocabulary_size)  # shape [num_samples, H, W, vocabulary_size]
    dirac_x0 = rearrange(dirac_x0, "b h w c -> b c h w")  # shape [num_samples, vocabulary_size, H, W]

    # Determine x1 using Euler
    xt, dirac_xt = x0, dirac_x0

    t = t_min * torch.ones(num_samples, device=device, dtype=backbone_dtype)  # Initialize t to t_min
    h = 1.0 / euler_steps

    if num_classes > 0 and y is not None:
        y = y.to(device)

    t = _pad_like_x(t, dirac_xt)
    with torch.no_grad():
        while t.max() <= 1 - h:
            t_batch = t.squeeze()
            p1t = F.softmax(backbone(xt, t_batch, y, cfg_scale), dim=1)

            pt = dirac_xt + h * (p1t - dirac_xt) / (1 - t)

            will_unmask = xt == vocabulary_size - 1
            will_mask = torch.rand(xt.shape, device=device) < 1 - t.max() - h  # (B, D)
            will_mask = will_mask * (xt != vocabulary_size - 1)  # (B, D)
            xt[will_mask] = vocabulary_size - 1

            _xt = sample_p(pt)
            xt[will_unmask] = _xt[will_unmask]  # unmask first

            t += h

            if t.max() < 1 - h:
                xt[will_mask] = vocabulary_size - 1  # mask later
            if t.max() >= 1 - h:
                xt = argmax_p(pt=pt, xt=xt, mask_token_id=vocabulary_size - 1)

            xt = sample_p(pt)
            dirac_xt = F.one_hot(xt, vocabulary_size)  # shape [num_samples, H, W, vocabulary_size]
            dirac_xt = rearrange(dirac_xt, "b h w c -> b c h w")  # shape [num_samples, vocabulary_size, H, W]

    xt = rearrange(xt, "b h w -> b (h w)")

    images = _decode_indices(autoencoder, xt)

    return images


def _ode_function_purrception(
    t,
    x,
    backbone,
    y,
    num_samples,
    input_size,
    codebook,
    cfg_scale=1.0,
    eps=1e-6,
    temperature=1.0,
):
    backbone_dtype = next(backbone.parameters()).dtype
    t = t.to(dtype=backbone_dtype)

    x = rearrange(x, "b (c h w) -> b c h w", b=num_samples, h=input_size, w=input_size)
    t_batch = t.expand(num_samples)
    mu = backbone(x, t_batch, y, cfg_scale)

    tau = max(temperature, eps)
    mu = mu / tau
    mu = F.softmax(mu, dim=1)
    mu = rearrange(mu, "b c h w -> (b h w) c")

    codebook = codebook.to(dtype=mu.dtype)
    mu = mu @ codebook  # (B * H * W, C) * (C, 4) --> (B * H * W, 4)

    mu = rearrange(mu, "(b h w) c -> b (c h w)", b=num_samples, h=input_size, w=input_size)
    x = rearrange(x, "b c h w -> b (c h w)")

    return (mu - x) / (1 - t + eps)


def sample_purrception(
    num_samples,
    autoencoder,
    backbone,
    backbone_config,
    ode_method,
    ode_steps,
    cfg_scale,
    y,
    device="cuda:0",
    temperature=1.0,
    atol=1e-9,
    rtol=1e-7,
    **kargs,
):
    input_size = backbone_config["input_size"]
    num_classes = backbone_config["num_classes"]
    in_channels = backbone_config["in_channels"]

    if num_classes > 0 and y is None:
        y = torch.randint(0, num_classes, (num_samples,))

    backbone_dtype = next(backbone.parameters()).dtype
    codebook = _get_codebook(autoencoder, device, backbone_dtype)
    print(f"Codebook shape: {codebook.shape}")
    K, d = codebook.shape

    x0 = torch.randn(num_samples, in_channels, input_size, input_size, dtype=backbone_dtype).to(device)
    t = torch.linspace(0, 1, ode_steps, dtype=backbone_dtype).to(device)

    if num_classes > 0 and y is not None:
        y = y.to(device)

    x0 = rearrange(x0, "b c h w -> b (c h w)")
    ode_func = partial(
        _ode_function_purrception,
        y=y,
        backbone=backbone,
        num_samples=num_samples,
        input_size=input_size,
        cfg_scale=cfg_scale,
        codebook=codebook,
        temperature=temperature,
    )

    with torch.no_grad():
        trajectory = odeint(func=ode_func, y0=x0, t=t, method=ode_method, atol=atol, rtol=rtol, adjoint_params=())

        x1 = trajectory[-1]
        x1 = rearrange(x1, "b (c h w) -> b c h w", b=num_samples, h=input_size, w=input_size)

        if _is_llamagen_tokenizer(autoencoder):
            original_dtype = autoencoder.quantize.embedding.weight.dtype
            autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                dtype=backbone_dtype
            )
            try:
                x1_q, _, _ = autoencoder.quantize(x1)
            finally:
                autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                    dtype=original_dtype
                )
        else:
            original_dtype = autoencoder.quantize.embedding.weight.dtype
            autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                dtype=backbone_dtype
            )
            try:
                x1_q, _, _ = autoencoder.quantize(x1)
            finally:
                autoencoder.quantize.embedding.weight.data = autoencoder.quantize.embedding.weight.data.to(
                    dtype=original_dtype
                )

        x1_q = x1_q.to(dtype=torch.float32)
        images = _decode_embeddings(autoencoder, x1_q)

        return images


def main():
    pass


if __name__ == "__main__":
    main()
