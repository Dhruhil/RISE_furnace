"""
3D Fourier Neural Operator for the heat-treatment problem.

Input:  (batch, 8, Gx, Gy, Gz)
        Channels: [T, T_set, region_id, time, is_heater, kappa, Cp, rho]
Output: (batch, 1, Gx, Gy, Gz)
        Channel: normalised T_next (full field, not delta)

Two backends live here. The official PhysicsNeMo FNO is the one
that produced the numbers reported in the thesis; the in-house
fallback exists so the model still runs on machines where
PhysicsNeMo isn't installed (laptop debug runs, mainly).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from pathlib import Path


# Try to use NVIDIA PhysicsNeMo's official 3D FNO first. The
# fallback below is a faithful, simpler reimplementation of the
# same architecture, intended for environments where PhysicsNeMo
# isn't available.
try:
    from physicsnemo.models.fno import FNO as _PhysicsNeMoFNO
    PHYSICSNEMO_FNO = True
    print("[INFO] Using official NVIDIA PhysicsNeMo 3D FNO")
except ImportError:
    PHYSICSNEMO_FNO = False
    print("[INFO] physicsnemo FNO not available — using fallback 3D FNO")


class _SpectralConv3d(nn.Module):
    """
    Single 3D Fourier layer. Takes the input tensor into the
    Fourier domain via an rFFT, multiplies by learnable complex
    weights on the truncated low-frequency block, and brings it
    back into physical space with an inverse rFFT.

    Truncating the high-frequency modes is the whole point — it
    keeps the parameter count bounded and naturally encodes a
    smoothness prior, which suits the slowly-varying temperature
    fields in this problem.
    """

    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes  # [mx, my, mz]
        # 1/(in*out) initialisation matches the original FNO paper
        # and keeps activations from blowing up at deeper layers.
        scale = 1 / (in_ch * out_ch)
        self.weights = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, *modes, dtype=torch.cfloat))

    def forward(self, x):
        B, C, Nx, Ny, Nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Clip the requested mode counts to whatever the input grid
        # can actually support. This matters when the inference grid
        # is smaller than the training grid — the spectral layer
        # still works, just over fewer modes.
        mx, my, mz = self.modes
        mx = min(mx, Nx // 2 + 1)
        my = min(my, Ny // 2 + 1)
        mz = min(mz, Nz // 2 + 1)

        # Output spectrum starts at zero everywhere; only the
        # truncated low-frequency block gets filled in.
        out_ft = torch.zeros(B, self.weights.shape[1],
                             Nx, Ny, Nz // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :mx, :my, :mz] = torch.einsum(
            "bcxyz,coxyz->boxyz",
            x_ft[:, :, :mx, :my, :mz],
            self.weights[:, :, :mx, :my, :mz])
        return torch.fft.irfftn(out_ft, s=[Nx, Ny, Nz], dim=[-3, -2, -1])


class _FNOBlock3d(nn.Module):
    """
    One FNO block: a spectral 3D conv in parallel with a pointwise
    1x1x1 conv (the "skip" path), then InstanceNorm + GELU.

    The pointwise conv lets the block represent local features that
    the truncated Fourier modes can't easily capture, so the two
    paths complement each other.
    """

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = _SpectralConv3d(width, width, modes)
        self.linear = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        return nn.functional.gelu(
            self.norm(self.spectral(x) + self.linear(x)))


class _FallbackFNO3d(nn.Module):
    """
    Stand-in 3D FNO for environments without PhysicsNeMo.

    Lift -> N spectral blocks (with residuals) -> small decoder MLP.
    Same overall topology as the official version, just leaner — the
    output should match the PhysicsNeMo backend up to numerical
    noise on the same training data.
    """

    def __init__(self, in_ch, out_ch, modes, width, n_layers,
                 dec_layers, dec_size):
        super().__init__()
        # Lift the 8-channel input into the latent width
        self.lift = nn.Conv3d(in_ch, width, 1)

        # Stack of spectral blocks — the "process" part of the
        # lift-process-project pattern.
        self.blocks = nn.ModuleList(
            [_FNOBlock3d(width, modes) for _ in range(n_layers)])

        # Per-voxel decoder built from 1x1x1 convs — projects the
        # latent width back down to the output channels.
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
            # Residual around each block — the FNO's deep-stack
            # stability story leans heavily on these skips.
            x = block(x) + x
        return self.decoder(x)


class HeatTreatmentFNO3D(nn.Module):
    """
    Wrapper that picks the PhysicsNeMo FNO when available, or the
    in-house fallback otherwise. Either way the public interface
    (forward / save / load) stays identical, so the training and
    rollout scripts don't care which backend is running underneath.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if PHYSICSNEMO_FNO:
            # Official PhysicsNeMo 3D FNO — this is what produced
            # the numbers reported in the thesis. padding=4 gives
            # the spectral layers a little room around the edges of
            # the voxel grid, which helps with boundary artefacts.
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
            # Fallback path — keeps things working on machines
            # without PhysicsNeMo (e.g., a quick laptop debug run).
            self.fno = _FallbackFNO3d(
                cfg.fno_in_channels, cfg.fno_out_channels,
                cfg.fno_modes, cfg.fno_latent, cfg.fno_layers,
                cfg.fno_decoder_layers, cfg.fno_decoder_layer_size)
            self._backend = "fallback"

        # Quick parameter count printout — useful sanity check after
        # changing fno_latent or fno_layers.
        n_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  HeatTreatmentFNO3D [{self._backend}]")
        print(f"    in={cfg.fno_in_channels} out={cfg.fno_out_channels} "
              f"modes={cfg.fno_modes} layers={cfg.fno_layers} latent={cfg.fno_latent}")
        print(f"    Grid: {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
        print(f"    Trainable parameters: {n_p:,}")

    def forward(self, x):
        # Single-pass call straight through the underlying FNO. The
        # input shape (B, 8, Gx, Gy, Gz) and output shape
        # (B, 1, Gx, Gy, Gz) are documented at the module top.
        return self.fno(x)

    def save(self, path, epoch, opt_state=None, sched_state=None, metrics=None):
        """
        Dump the full training state to a single .pt file so
        training can resume from exactly where it left off if a
        SLURM job gets killed mid-run.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "model_state": self.state_dict(),
                     "optimizer_state": opt_state, "scheduler_state": sched_state,
                     "metrics": metrics or {}, "backend": self._backend}, path)

    @classmethod
    def load(cls, path, cfg, device="cpu"):
        """
        Restore the model from a checkpoint produced by .save().
        The cfg passed in here should match the one used at training
        time — backend mismatches (PhysicsNeMo vs fallback) will
        raise a load_state_dict error, which is the desired
        behaviour rather than silently using mismatched weights.
        """
        ckpt = torch.load(path, map_location=device)
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        return model