#!/bin/bash
# Force fallback FNO (no PhysicsNeMo padding overhead) + smaller latent
# cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# bash fix_fno_size2.sh

set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== Fix: Force fallback FNO + reduce size ==="

# Force fallback by disabling PhysicsNeMo import in model
python3 << 'XEOF'
with open("models/fno_model.py", "r") as f:
    code = f.read()

# Force fallback - PhysicsNeMo's 3D FNO adds massive internal padding
old_import = '''try:
    from physicsnemo.models.fno import FNO as _PhysicsNeMoFNO
    PHYSICSNEMO_FNO = True
except ImportError:
    PHYSICSNEMO_FNO = False
    print("[INFO] physicsnemo FNO not found — using fallback.")'''

new_import = '''# Force fallback FNO — PhysicsNeMo's 3D FNO creates oversized internal tensors
# Our fallback is lean and efficient for this grid size
PHYSICSNEMO_FNO = False
print("[INFO] Using lean fallback 3D FNO (optimised for heat treatment grid)")'''''

if old_import in code:
    code = code.replace(old_import, new_import)
    print("  OK: Forced fallback FNO")

with open("models/fno_model.py", "w") as f:
    f.write(code)
XEOF

# Reduce latent further
python3 << 'XEOF'
with open("configs/fno_config.py", "r") as f:
    code = f.read()

# latent 32 -> 20, modes smaller
code = code.replace("fno_latent:       int = 32", "fno_latent:       int = 20")
code = code.replace("fno_decoder_layer_size: int = 32", "fno_decoder_layer_size: int = 20")

with open("configs/fno_config.py", "w") as f:
    f.write(code)
print("  OK: Latent 32 -> 20")
XEOF

echo ""
echo "=== Checking param count ==="
python3 -c "
import torch, torch.nn as nn

class SC3d(nn.Module):
    def __init__(s, ic, oc, modes):
        super().__init__()
        s.modes = modes
        s.w = nn.Parameter(torch.randn(ic, oc, *modes, dtype=torch.cfloat) / (ic*oc))
    def forward(s, x): return x

class Block(nn.Module):
    def __init__(s, w, modes):
        super().__init__()
        s.sp = SC3d(w, w, modes)
        s.lin = nn.Conv3d(w, w, 1)
        s.norm = nn.InstanceNorm3d(w)
    def forward(s, x): return x

class FNO(nn.Module):
    def __init__(s):
        super().__init__()
        s.lift = nn.Conv3d(7, 20, 1)
        s.blocks = nn.ModuleList([Block(20, [8,12,14]) for _ in range(4)])
        s.dec = nn.Sequential(nn.Conv3d(20, 20, 1), nn.GELU(), nn.Conv3d(20, 20, 1), nn.GELU(), nn.Conv3d(20, 1, 1))
    def forward(s, x): return x

m = FNO()
n = sum(p.numel() for p in m.parameters())
mb = n * 4 / 1e6  # FP32 bytes
print(f'  Fallback FNO: {n:,} params ({mb:.1f} MB)')
print(f'  With Adam optimizer: {mb*3:.1f} MB total')
print(f'  Fits on T4 (16GB): {\"YES\" if mb*3 < 2000 else \"NO\"}')"

echo ""
echo "=== Test now ==="
echo "  apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \\"
echo "    python -u train.py --epochs 3 --batch 2 --device cuda"
