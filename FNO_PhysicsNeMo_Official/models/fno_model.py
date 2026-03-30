"""
3D Fourier Neural Operator for heat treatment.
Input:  (batch, 8, Gx, Gy, Gz)  — [T, T_set, region_id, time, is_heater, kappa, Cp, rho]
Output: (batch, 1, Gx, Gy, Gz)  — normalised T_next
"""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path

# Force fallback FNO — PhysicsNeMo's 3D FNO creates oversized internal tensors
# Our fallback is lean and efficient for this grid size
PHYSICSNEMO_FNO = False
print("[INFO] Using lean fallback 3D FNO (optimised for heat treatment grid)")


class _SpectralConv3d(nn.Module):
    """3D Fourier layer."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes  # [mx, my, mz]
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, *modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Nx, Ny, Nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        mx, my, mz = self.modes
        mx = min(mx, Nx // 2 + 1)
        my = min(my, Ny // 2 + 1)
        mz = min(mz, Nz // 2 + 1)
        out_ft = torch.zeros(B, self.weights.shape[1],
                             Nx, Ny, Nz // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,coxyz->boxyz",
            x_ft[:, :, :mx, :my, :mz],
            self.weights[:, :, :mx, :my, :mz])
        return torch.fft.irfftn(out_ft, s=[Nx, Ny, Nz], dim=[-3, -2, -1])


class _FNOBlock3d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = _SpectralConv3d(width, width, modes)
        self.linear = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        return nn.functional.gelu(
            self.norm(self.spectral(x) + self.linear(x)))


class _FallbackFNO3d(nn.Module):
    def __init__(self, in_ch, out_ch, modes, width, n_layers,
                 dec_layers, dec_size):
        super().__init__()
        self.lift = nn.Conv3d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            [_FNOBlock3d(width, modes) for _ in range(n_layers)])
        layers = []
        prev = width
        for _ in range(dec_layers):
            layers += [nn.Conv3d(prev, dec_size, 1), nn.GELU()]
            prev = dec_size
        layers.append(nn.Conv3d(prev, out_ch, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.lift(x)
        for block in self.blocks:
            x = block(x) + x
        return self.decoder(x)


class HeatTreatmentFNO3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if PHYSICSNEMO_FNO:
            self.fno = _PhysicsNeMoFNO(
                in_channels=cfg.fno_in_channels,
                out_channels=cfg.fno_out_channels,
                num_fno_modes=cfg.fno_modes,
                num_fno_layers=cfg.fno_layers,
                latent_channels=cfg.fno_latent,
                decoder_layers=cfg.fno_decoder_layers,
                decoder_layer_size=cfg.fno_decoder_layer_size,
                dimension=3,
                padding=4,
            )
            self._backend = "physicsnemo"
        else:
            self.fno = _FallbackFNO3d(
                cfg.fno_in_channels, cfg.fno_out_channels,
                cfg.fno_modes, cfg.fno_latent, cfg.fno_layers,
                cfg.fno_decoder_layers, cfg.fno_decoder_layer_size)
            self._backend = "fallback"
        n_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  HeatTreatmentFNO3D [{self._backend}]")
        print(f"    in={cfg.fno_in_channels} out={cfg.fno_out_channels} "
              f"modes={cfg.fno_modes} layers={cfg.fno_layers} latent={cfg.fno_latent}")
        print(f"    Grid: {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
        print(f"    Trainable parameters: {n_p:,}")

    def forward(self, x):
        return self.fno(x)

    def save(self, path, epoch, opt_state=None, sched_state=None, metrics=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model_state": self.state_dict(),
                     "optimizer_state": opt_state, "scheduler_state": sched_state,
                     "metrics": metrics or {}, "backend": self._backend}, path)

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return model
