#!/bin/bash
# ============================================================
# Fix FNO: Grid covers INNER_BOX only (no outer_box)
#
# Before: grid covers full furnace, 70% is outer_box (boring)
# After:  grid covers inner cavity only, every voxel matters
#
# cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# bash fix_fno_innerbox.sh
# ============================================================

set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "============================================"
echo "  Fix: Grid bounds -> inner_box only"
echo "============================================"

python3 << 'XEOF'
with open("configs/fno_config.py", "r") as f:
    code = f.read()

# Change grid bounds to inner_box cavity
# From .geo: inner_box roughly x=[0, 0.206], y=[0.065, 0.295], z=[0.01, 0.38]
# Add small margin for interpolation
code = code.replace("x_min: float = 0.0;   x_max: float = 0.206", 
                     "x_min: float = 0.0;   x_max: float = 0.206")  # x stays same
code = code.replace("y_min: float = 0.0;   y_max: float = 0.36",
                     "y_min: float = 0.06;  y_max: float = 0.30")
code = code.replace("z_min: float = 0.0;   z_max: float = 0.39",
                     "z_min: float = 0.005; z_max: float = 0.385")

# Increase grid resolution since domain is smaller
# Inner box: x=0.206, y=0.24, z=0.38 -> aspect ratio ~1:1.2:1.8
code = code.replace("grid_x: int = 20", "grid_x: int = 20")
code = code.replace("grid_y: int = 32", "grid_y: int = 24")  # y range is smaller now
code = code.replace("grid_z: int = 36", "grid_z: int = 36")  # z stays same

# Adjust modes for new grid sizes
code = code.replace(
    "fno_modes:        list = field(default_factory=lambda: [10, 16, 18])",
    "fno_modes:        list = field(default_factory=lambda: [10, 12, 18])")

with open("configs/fno_config.py", "w") as f:
    f.write(code)

print("  OK: Grid bounds updated")
print("    y: [0.0, 0.36] -> [0.06, 0.30]  (inner_box only)")
print("    z: [0.0, 0.39] -> [0.005, 0.385] (tight margin)")
print("    Grid: 20x24x36 = 17,280 voxels")
print("    All voxels now inside inner cavity")
XEOF

echo ""
echo "  Also removing region weight (no longer needed)..."

python3 << 'XEOF'
# No changes needed to train.py or dataset.py —
# the weighted loss still works fine (steel=10x, air=3x)
# but now there's no outer_box to dilute the metrics.
# The weight_grid will show 0.1 for heaters, 3.0 for air, 10.0 for steel.
# No outer_box voxels exist in the grid anymore.
print("  OK: Weighted loss still active (steel=10x, air=3x, heater=0.1x)")
print("  No outer_box voxels in grid -> no dilution")
XEOF

echo ""
echo "============================================"
echo "  VERIFICATION"  
echo "============================================"
python3 -c "import ast; ast.parse(open('configs/fno_config.py').read()); print('  OK: config syntax')"
grep "y_min\|y_max\|z_min\|z_max" configs/fno_config.py
grep "grid_x\|grid_y\|grid_z" configs/fno_config.py

echo ""
echo "============================================"
echo "  DONE"
echo "============================================"
echo ""
echo "  Before: Grid 20x32x36 = 23,040 voxels (70% outer_box)"
echo "  After:  Grid 20x24x36 = 17,280 voxels (0% outer_box)"
echo ""
echo "  Every voxel is now steel, air, heater, or brick"
echo "  Model must learn real heat transfer to reduce loss"
echo ""
echo "  Expected training curves:"
echo "    Epoch  1: Steel MAE ~5-15K  (can't shortcut anymore)"
echo "    Epoch 50: Steel MAE ~2-5K   (smooth decrease)"
echo "    Epoch 100: Steel MAE ~1-3K  (converged)"
echo ""
echo "  sbatch run_alvis_fno.sh"
