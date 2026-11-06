"""Future-latent estimation with the frozen V-JEPA predictor (blueprint §5.3, Option A).

`anticipate` mirrors `AnticipativeWrapper.forward` from facebookresearch/vjepa2
(evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py) for num_steps=1,
but also returns the raw predictor output, so callers can check which feature space it lives in.
Feature extraction can use it to store encoder tokens and predicted tokens in one pass.
"""
from __future__ import annotations
import importlib
from pathlib import Path
import sys
import torch
from torch import nn


def anticipate(model, x, anticipation_times):
    """Encode an observed clip and predict the latent slice `anticipation_times` seconds after it ends.

    x: [B, 3, T, H, W] normalized frames that END at the observation cutoff.
    anticipation_times: [B] seconds. The predicted slice covers `tubelet_size` frames starting
    anticipation_time after the last observed slice, exactly as in the official wrapper.
    Returns (encoder_tokens, encoder_full, predicted_tokens, predicted_full):
    *_tokens use the last-layer width (model.encoder.embed_dim), as the official wrapper returns;
    *_full are the raw outputs (hierarchical encoders and distilled predictors are wider).
    """
    if getattr(model, "num_steps", 1) != 1 or getattr(model, "no_predictor", False) or getattr(model, "no_encoder", False):
        raise NotImplementedError("Only the official setting num_steps=1 with encoder and predictor is supported")
    encoder_full = model.encoder(x)
    B, N, D_full = encoder_full.shape
    embed_dim = model.encoder.embed_dim
    encoder_tokens = encoder_full[:, :, -embed_dim:] if D_full > embed_dim else encoder_full
    ctxt_positions = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
    steps = (anticipation_times * model.frames_per_second / model.tubelet_size).to(torch.int64).to(x.device)
    skip_positions = N + int(model.grid_size ** 2) * steps
    n_pred = int(model.grid_size ** 2 * (model.num_output_frames // model.tubelet_size))
    tgt_positions = torch.arange(n_pred, device=x.device).unsqueeze(0).repeat(B, 1) + skip_positions.unsqueeze(1)
    out = model.predictor(encoder_full, masks_x=ctxt_positions, masks_y=tgt_positions)
    predicted_full = out[0] if isinstance(out, tuple) else out
    predicted_tokens = predicted_full[:, :, -embed_dim:] if predicted_full.size(-1) != embed_dim else predicted_full
    return encoder_tokens, encoder_full, predicted_tokens, predicted_full


def same_feature_space(encoder_full, predicted_full, embed_dim):
    """True when predicted tokens can be compared with encoder tokens.

    V-JEPA 2 predictors regress the (EMA) encoder's own features. Distilled V-JEPA 2.1 students
    (ViT-B/L 'dist_vitG') predict the ViT-G teacher's features instead (e.g. 1664 vs 1024 dims);
    slicing the last embed_dim channels then gives a vector of the right size in a different space.
    """
    return predicted_full.size(-1) in {embed_dim, encoder_full.size(-1)}


def load_vjepa_model(vjepa_root, config: dict, checkpoint, frames_per_clip, frames_per_second, resolution):
    """Build the frozen official wrapper from an eval config (e.g. configs/eval_2_1/vitl-384/ek100.yaml)."""
    root = str(Path(vjepa_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    kwargs = config["model_kwargs"]
    module = importlib.import_module(kwargs["module_name"])
    model = module.init_module(frames_per_clip=frames_per_clip, frames_per_second=frames_per_second,
                               resolution=resolution, checkpoint=str(checkpoint),
                               model_kwargs=kwargs["pretrain_kwargs"], wrapper_kwargs=kwargs["wrapper_kwargs"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class _DummyEncoder(nn.Module):
    def __init__(self, embed_dim, patch_size, tubelet_size):
        super().__init__()
        self.embed_dim, self.patch_size, self.tubelet_size = embed_dim, patch_size, tubelet_size
        self.patchify = nn.Conv3d(3, embed_dim, (tubelet_size, patch_size, patch_size),
                                  stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x):
        return self.patchify(x).flatten(2).transpose(1, 2)  # [B, T'*H'*W', D], time-major like V-JEPA


class _DummyPredictor(nn.Module):
    def __init__(self, embed_dim, max_positions=8192):
        super().__init__()
        self.position = nn.Embedding(max_positions, embed_dim)
        self.mix = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, masks_x, masks_y):
        context = x.mean(dim=1, keepdim=True)
        return self.mix(context + self.position(masks_y.clamp(max=self.position.num_embeddings - 1)))


class DummyAnticipativeModel(nn.Module):
    """Tiny random stand-in with the AnticipativeWrapper interface, for tests and dry runs on CPU."""

    def __init__(self, resolution=32, patch_size=16, tubelet_size=2, embed_dim=32,
                 frames_per_second=8, num_output_frames=2):
        super().__init__()
        self.encoder = _DummyEncoder(embed_dim, patch_size, tubelet_size)
        self.predictor = _DummyPredictor(embed_dim)
        self.grid_size = resolution // patch_size
        self.tubelet_size, self.frames_per_second = tubelet_size, frames_per_second
        self.num_output_frames, self.num_steps, self.embed_dim = num_output_frames, 1, embed_dim

    def forward(self, x, anticipation_times):
        encoder_tokens, _, predicted_tokens, _ = anticipate(self, x, anticipation_times)
        return torch.cat([encoder_tokens, predicted_tokens], dim=1)
