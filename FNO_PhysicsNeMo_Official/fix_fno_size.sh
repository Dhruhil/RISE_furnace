#!/bin/bash
# Fix oversized 3D FNO model (402M params -> ~2-5M params)
# Problem: 19 channels + 64 latent + 3D = massive PhysicsNeMo FNO
# Solution: smaller grid, fewer modes, fewer input channels
#
# cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# bash fix_fno_size.sh

set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== Fixing oversized 3D FNO ==="

python3 << 'XEOF'
with open("configs/fno_config.py", "r") as f:
    code = f.read()

# Reduce grid resolution: 24x40x44 -> 16x24x28
code = code.replace("grid_x: int = 24", "grid_x: int = 16")
code = code.replace("grid_y: int = 40", "grid_y: int = 24")
code = code.replace("grid_z: int = 44", "grid_z: int = 28")

# Reduce modes to fit smaller grid (modes must be < grid_size/2)
code = code.replace(
    "fno_modes:        list = field(default_factory=lambda: [12, 16, 16])",
    "fno_modes:        list = field(default_factory=lambda: [8, 12, 14])")

# Reduce latent: 64 -> 32 (3D FNO internal tensors are huge)
code = code.replace("fno_latent:       int = 64", "fno_latent:       int = 32")

# Reduce decoder
code = code.replace("fno_decoder_layer_size: int = 64", "fno_decoder_layer_size: int = 32")

# Reduce input channels: 19 -> 7
# Instead of 12 one-hot region masks, use 1 region_id/11 channel
# [T_norm, T_set_norm, region_id/11, time, is_heater, kappa/100, rho/10000]
code = code.replace("fno_in_channels:  int = 19", "fno_in_channels:  int = 7")

# Comment explaining the change
code = code.replace(
    "# Input channels: T_norm, T_set_norm, region_mask (12 binary), time,\n"
    "    #                 is_heater, kappa, Cp, rho = 1+1+12+1+1+3 = 19",
    "# Input channels: T_norm, T_set_norm, region_id/11, time,\n"
    "    #                 is_heater, kappa/100, rho/10000 = 7\n"
    "    # (single region_id channel instead of 12 one-hot masks)")

with open("configs/fno_config.py", "w") as f:
    f.write(code)
print("  OK: Config updated (grid 16x24x28, 7 channels, 32 latent)")
XEOF

# Now update dataset to produce 7 channels instead of 19
python3 << 'XEOF'
with open("data/dataset.py", "r") as f:
    code = f.read()

# Replace the input channel construction in __getitem__
old_channels = '''        # Build input: (19, Gx, Gy, Gz)
        Gx, Gy, Gz = self.grid_shape
        x = np.zeros((19, Gx, Gy, Gz), dtype=np.float32)
        x[0] = T_norm                                    # T_current
        x[1] = Tset_norm                                 # T_set (scalar broadcast)
        x[2:14] = fields["region_onehot"].transpose(3, 0, 1, 2)  # 12 region masks
        x[14] = t_norm                                   # time
        x[15] = fields["is_heater"].squeeze(-1)          # is_heater
        x[16] = fields["kappa"].squeeze(-1)              # kappa/100
        x[17] = fields["Cp"].squeeze(-1)                 # Cp/1000
        x[18] = fields["rho"].squeeze(-1)                # rho/10000'''

new_channels = '''        # Build input: (7, Gx, Gy, Gz)
        Gx, Gy, Gz = self.grid_shape
        x = np.zeros((7, Gx, Gy, Gz), dtype=np.float32)
        x[0] = T_norm                                    # T_current
        x[1] = Tset_norm                                 # T_set (scalar broadcast)
        x[2] = fields["region_id"].squeeze(-1)           # region_id / 11
        x[3] = t_norm                                    # time
        x[4] = fields["is_heater"].squeeze(-1)           # is_heater
        x[5] = fields["kappa"].squeeze(-1)               # kappa/100
        x[6] = fields["rho"].squeeze(-1)                 # rho/10000'''

if old_channels in code:
    code = code.replace(old_channels, new_channels)
    print("  OK: Dataset channels 19 -> 7")

# Add region_id field to the static interpolation
# Replace region_onehot with region_id in the interpolation block
old_interp = '''            for ch_name, ch_data in [
                ("region_onehot", sim["region_onehot"]),
                ("is_heater", sim["is_heater"][:, None]),
                ("kappa", sim["kappa"][:, None] / 100.0),
                ("Cp", sim["Cp"][:, None] / 1000.0),
                ("rho", sim["rho"][:, None] / 10000.0),
            ]:'''

new_interp = '''            # Region ID as single float channel (not one-hot)
            region_ids_float = np.zeros((sim["total_cells"], 1), dtype=np.float32)
            for j in range(sim["total_cells"]):
                # Find which region this cell belongs to by checking onehot
                region_ids_float[j, 0] = np.argmax(sim["region_onehot"][j]) / 11.0

            for ch_name, ch_data in [
                ("region_id", region_ids_float),
                ("is_heater", sim["is_heater"][:, None]),
                ("kappa", sim["kappa"][:, None] / 100.0),
                ("rho", sim["rho"][:, None] / 10000.0),
            ]:'''

if old_interp in code:
    code = code.replace(old_interp, new_interp)
    print("  OK: Interpolation uses region_id instead of one-hot")

# Also update the docstring
code = code.replace(
    '''    Channels (19 total):
      [0]     T_norm           current temperature
      [1]     T_set_norm       furnace setpoint
      [2-13]  region_mask      12 binary channels (one-hot per region)
      [14]    time_norm        t / 4000
      [15]    is_heater        binary
      [16]    kappa_norm       thermal conductivity / 100
      [17]    Cp_norm          specific heat / 1000
      [18]    rho_norm         density / 10000''',
    '''    Channels (7 total):
      [0]  T_norm        current temperature
      [1]  T_set_norm    furnace setpoint
      [2]  region_id/11  region encoding (0=steel, 11=outer_box)
      [3]  time/4000     normalised time
      [4]  is_heater     binary heater flag
      [5]  kappa/100     thermal conductivity
      [6]  rho/10000     density''')
print("  OK: Docstring updated")

with open("data/dataset.py", "w") as f:
    f.write(code)
XEOF

# Update physics loss to use correct channel indices
python3 << 'XEOF'
with open("train.py", "r") as f:
    code = f.read()

# Fix channel index for is_heater (was [15], now [4])
code = code.replace("is_heater = x[:, 15]", "is_heater = x[:, 4]")

# Fix assert
code = code.replace("assert x.shape[1] == 19", "assert x.shape[1] == 7")

with open("train.py", "w") as f:
    f.write(code)
print("  OK: train.py channel indices updated")
XEOF

echo ""
echo "=== VERIFICATION ==="
grep "fno_in_channels" configs/fno_config.py
grep "fno_latent" configs/fno_config.py
grep "grid_x" configs/fno_config.py
grep "fno_modes" configs/fno_config.py
grep "x\[0\] = T_norm" data/dataset.py
grep "is_heater = x\[:, 4\]" train.py

echo ""
echo "=== DONE ==="
echo ""
echo "  Before: 19 channels, 64 latent, 24x40x44 grid = 402M params"
echo "  After:  7 channels, 32 latent, 16x24x28 grid = ~2-5M params"
echo ""
echo "  Test: apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \\"
echo "          python -u train.py --epochs 3 --batch 2 --device cuda"
