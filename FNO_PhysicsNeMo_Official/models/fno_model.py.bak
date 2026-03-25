"""
Heat Treatment FNO — NVIDIA PhysicsNeMo 25.06.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Uses the official PhysicsNeMo 1D FNO for temperature prediction.
Falls back to pure-PyTorch FNO if PhysicsNeMo is not available.

Architecture:
    Input:  (batch, 4, n_cells)  — T_now + T_set + region_id + time
    Output: (batch, 1, n_cells)  — T_next prediction
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

try:
    from physicsnemo.models.fno import FNO as _PhysicsNeMoFNO
    PHYSICSNEMO_FNO = True
except ImportError:
    PHYSICSNEMO_FNO = False
    print("[INFO] physicsnemo.models.fno not found — using fallback FNO.")


# ─────────────────────────────────────────────────────────────────────
# Fallback: pure PyTorch 1D FNO
# ─────────────────────────────────────────────────────────────────────

class _SpectralConv1d(nn.Module):
    """1D Fourier layer: spectral convolution in frequency domain."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.weights.shape[1], N // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        m = min(self.modes, N // 2 + 1)
        out_ft[:, :, :m] = torch.einsum(
            "bci,iom->bom", x_ft[:, :, :m], self.weights[:, :, :m]
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class _FNOBlock1d(nn.Module):
    """Single FNO block: spectral conv + linear skip + norm."""
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = _SpectralConv1d(width, width, modes)
        self.linear   = nn.Conv1d(width, width, 1)
        self.norm     = nn.InstanceNorm1d(width)

    def forward(self, x):
        return nn.functional.gelu(
            self.norm(self.spectral(x) + self.linear(x))
        )


class _FallbackFNO1d(nn.Module):
    """Pure PyTorch 1D FNO — no PhysicsNeMo dependency."""
    def __init__(self, in_ch, out_ch, modes, width, n_layers,
                 decoder_layers, decoder_size):
        super().__init__()
        self.lift   = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [_FNOBlock1d(width, modes) for _ in range(n_layers)]
        )
        layers = []
        prev = width
        for _ in range(decoder_layers):
            layers += [nn.Conv1d(prev, decoder_size, 1), nn.GELU()]
            prev = decoder_size
        layers.append(nn.Conv1d(prev, out_ch, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x) + x   # residual connection
        return self.decoder(x)


# ─────────────────────────────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────────────────────────────

class HeatTreatmentFNO(nn.Module):
    """
    1D Fourier Neural Operator for heat treatment temperature prediction.

    Wraps the official NVIDIA PhysicsNeMo FNO with automatic fallback
    to a pure PyTorch implementation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if PHYSICSNEMO_FNO:
            self.fno = _PhysicsNeMoFNO(
                in_channels        = cfg.fno_in_channels,
                out_channels       = cfg.fno_out_channels,
                num_fno_modes      = [cfg.fno_modes],
                num_fno_layers     = cfg.fno_layers,
                latent_channels    = cfg.fno_latent,
                decoder_layers     = cfg.fno_decoder_layers,
                decoder_layer_size = cfg.fno_decoder_layer_size,
                dimension          = 1,
                padding            = 8,
            )
            self._backend = "physicsnemo"
        else:
            self.fno = _FallbackFNO1d(
                in_ch          = cfg.fno_in_channels,
                out_ch         = cfg.fno_out_channels,
                modes          = cfg.fno_modes,
                width          = cfg.fno_latent,
                n_layers       = cfg.fno_layers,
                decoder_layers = cfg.fno_decoder_layers,
                decoder_size   = cfg.fno_decoder_layer_size,
            )
            self._backend = "fallback"

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  HeatTreatmentFNO [{self._backend}]")
        print(f"    in={cfg.fno_in_channels}  out={cfg.fno_out_channels}  "
              f"modes={cfg.fno_modes}  layers={cfg.fno_layers}  "
              f"latent={cfg.fno_latent}")
        print(f"    Trainable parameters: {n_params:,}")

    def forward(self, x):
        """x: (batch, in_channels, n_cells) → (batch, 1, n_cells)"""
        return self.fno(x)

    def save(self, path, epoch, optimizer_state=None,
             scheduler_state=None, metrics=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch":           epoch,
            "model_state":     self.state_dict(),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "metrics":         metrics or {},
            "backend":         self._backend,
            "model_cfg": {
                "fno_in_channels":  self.cfg.fno_in_channels,
                "fno_out_channels": self.cfg.fno_out_channels,
                "fno_modes":        self.cfg.fno_modes,
                "fno_layers":       self.cfg.fno_layers,
                "fno_latent":       self.cfg.fno_latent,
            },
        }, path)
        mae = metrics.get("mae", None) if metrics else None
        mae_str = f"  val_MAE={mae:.3f}K" if mae else ""
        print(f"  Checkpoint saved -> {path}  (epoch {epoch}){mae_str}")

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        ckpt  = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        e = ckpt.get("epoch", "?")
        m = ckpt.get("metrics", {})
        mae_str = f"  val_MAE={m['mae']:.3f}K" if "mae" in m else ""
        print(f"  FNO loaded <- {path}  (epoch {e}){mae_str}")
        return model
