"""
PINN Model — PhysicsNeMo + PyTorch fallback.
Master's Thesis: Digital Twin Modeling of Heat Treatment in Cast Metals

Architecture:
    Input:  (batch, 6) = [x, y, z, t, T_set, region_id]  (normalised)
    Output: (batch, 1) = [T]  (normalised)

Physics: heat equation PDE residual computed via automatic differentiation.
    ρ·Cp·∂T/∂t = κ·∇²T
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

try:
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch
    from physicsnemo.sym.models.activation import Activation
    PHYSICSNEMO_SYM = True
except ImportError:
    PHYSICSNEMO_SYM = False
    print("[INFO] physicsnemo.sym not found — using PyTorch fallback PINN.")


# ─────────────────────────────────────────────────────────────────────
# SIREN layer (sinusoidal activation — better than tanh for PINNs)
# ─────────────────────────────────────────────────────────────────────

class SirenLayer(nn.Module):
    """Sinusoidal representation network layer (Sitzmann et al., 2020)."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                import math
                self.linear.weight.uniform_(
                    -math.sqrt(6 / in_features) / omega_0,
                     math.sqrt(6 / in_features) / omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FallbackPINN(nn.Module):
    """Pure PyTorch PINN with SIREN or Tanh activation."""
    def __init__(self, n_in, n_out, width, n_layers, activation="siren", omega_0=30.0):
        super().__init__()
        layers = []
        if activation == "siren":
            layers.append(SirenLayer(n_in, width, is_first=True, omega_0=omega_0))
            for _ in range(n_layers - 1):
                layers.append(SirenLayer(width, width, omega_0=omega_0))
        else:
            layers.append(nn.Linear(n_in, width))
            layers.append(nn.Tanh())
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.Tanh())
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
# Main PINN model
# ─────────────────────────────────────────────────────────────────────

class HeatTreatmentPINN(nn.Module):
    """
    Physics-Informed Neural Network for heat treatment.

    Input:  (batch, 6) = [x_n, y_n, z_n, t_n, Tset_n, rid_n]
    Output: (batch, 1) = [T_n]  (normalised temperature)

    Uses PhysicsNeMo FullyConnected if available, else pure PyTorch.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if PHYSICSNEMO_SYM:
            try:
                from physicsnemo.sym.key import Key
                self.net = FullyConnectedArch(
                    input_keys=[Key(f"in_{i}") for i in range(cfg.n_inputs)],
                    output_keys=[Key("T")],
                    layer_size=cfg.hidden_width,
                    nr_layers=cfg.n_hidden_layers,
                    activation_fn=(Activation.SIREN if cfg.activation == "siren"
                                   else Activation.TANH),
                )
                self._backend = "physicsnemo_sym"
            except Exception as e:
                print(f"[WARN] PhysicsNeMo Sym FullyConnected failed: {e}")
                print("[INFO] Falling back to PyTorch PINN.")
                self.net = FallbackPINN(
                    cfg.n_inputs, cfg.n_outputs, cfg.hidden_width,
                    cfg.n_hidden_layers, cfg.activation, cfg.omega_0)
                self._backend = "fallback"
        else:
            self.net = FallbackPINN(
                cfg.n_inputs, cfg.n_outputs, cfg.hidden_width,
                cfg.n_hidden_layers, cfg.activation, cfg.omega_0)
            self._backend = "fallback"

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  HeatTreatmentPINN [{self._backend}]")
        print(f"    in={cfg.n_inputs}  out={cfg.n_outputs}  "
              f"width={cfg.hidden_width}  layers={cfg.n_hidden_layers}  "
              f"activation={cfg.activation}")
        print(f"    Trainable parameters: {n_params:,}")

    def forward(self, x):
        """x: (batch, 6) → (batch, 1)"""
        if self._backend == "physicsnemo_sym":
            # PhysicsNeMo expects dict input
            inp = {f"in_{i}": x[:, i:i+1] for i in range(x.shape[1])}
            out = self.net(inp)
            return out["T"]
        else:
            return self.net(x)

    def save(self, path, epoch, optimizer_state=None,
             scheduler_state=None, metrics=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": self.state_dict(),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "metrics": metrics or {},
            "backend": self._backend,
        }, path)
        mae = metrics.get("mae", None) if metrics else None
        mae_str = f"  val_MAE={mae:.3f}K" if mae else ""
        print(f"  Checkpoint saved -> {path}  (epoch {epoch}){mae_str}")

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        e = ckpt.get("epoch", "?")
        m = ckpt.get("metrics", {})
        mae_str = f"  val_MAE={m['mae']:.3f}K" if "mae" in m else ""
        print(f"  PINN loaded <- {path}  (epoch {e}){mae_str}")
        return model
