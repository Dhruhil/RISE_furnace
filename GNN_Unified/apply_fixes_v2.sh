#!/bin/bash
# ============================================================
# GNN_Unified fix v2 — per-region material properties
# ============================================================
set -u
cd /mimer/NOBACKUP/groups/revar/GNN_Unified || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
BK=backup_v2_${STAMP}
mkdir -p "$BK"

echo "============================================================"
echo "  GNN_Unified v2 — per-region material properties"
echo "  Backup dir: $BK"
echo "============================================================"

# ── Back up files we will touch ─────────────────────────────
for f in configs/base_config.py data/dataset_unified.py; do
    cp "$f" "$BK/$(basename "$f").bak"
done
echo "[1/5] Backups created."

# ── STEP 2: Add REGION_MATERIALS table to base_config.py ────
python3 << 'PYEOF'
import pathlib, re
p = pathlib.Path("configs/base_config.py")
src = p.read_text()

if "REGION_MATERIALS" in src:
    print("  REGION_MATERIALS already present — skipping.")
else:
    block = '''
# ── Per-region material properties (from OpenFOAM thermophysicalProperties) ──
# Each region has its own (kappa [W/m.K], Cp [J/kg.K], rho [kg/m^3]).
# inner_box is a fluid (air); k ~ 0.05 used as low-conduction placeholder.
REGION_MATERIALS = {
    "steel_cylinder": {"kappa": 80.0,  "Cp": 450.0,  "rho": 7800.0},
    "inner_box":      {"kappa": 0.05,  "Cp": 1000.0, "rho": 1.2},     # air (fluid)
    "outer_box":      {"kappa": 15.0,  "Cp": 1000.0, "rho": 867.0},
    "brick_heater":   {"kappa": 8.0,   "Cp": 450.0,  "rho": 7800.0},
    "heater_1":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_2":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_3":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_4":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_5":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_6":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_7":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
    "heater_8":       {"kappa": 80.0,  "Cp": 450.0,  "rho": 8000.0},
}
'''
    # Insert right after the imports / before the first @dataclass
    m = re.search(r"^@dataclass", src, flags=re.MULTILINE)
    if m is None:
        print("  ERROR: could not locate @dataclass in base_config.py")
        raise SystemExit(1)
    src = src[:m.start()] + block + "\n" + src[m.start():]
    p.write_text(src)
    print("  REGION_MATERIALS table inserted.")
PYEOF
echo "[2/5] base_config.py updated."

# ── STEP 3: Patch dataset_unified.py to use per-region materials ──
python3 << 'PYEOF'
import pathlib, re
p = pathlib.Path("data/dataset_unified.py")
src = p.read_text()

# 3a. Import REGION_MATERIALS at top of file (once)
if "from configs.base_config import REGION_MATERIALS" not in src:
    # Insert after the last "import h5py" line or equivalent
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("import ") or ln.strip().startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, "from configs.base_config import REGION_MATERIALS\n")
    src = "".join(lines)
    print("  Added import of REGION_MATERIALS.")

# 3b. Replace the single-value kappa/Cp/rho node-feature block
# Old (3 lines around 304-306):
#     np.full(total, sim["kappa"] / 100.0, dtype=np.float32),    # [13] kappa
#     np.full(total, sim["Cp"] / 1000.0, dtype=np.float32),      # [14] Cp
#     np.full(total, sim["rho"] / 10000.0, dtype=np.float32),    # [15] rho
#
# New: build arrays indexed by region, using the per-node region_id list
old_pat = re.compile(
    r"np\.full\(total, sim\[\"kappa\"\] / 100\.0, dtype=np\.float32\),\s*#\s*\[13\][^\n]*\n"
    r"\s*np\.full\(total, sim\[\"Cp\"\] / 1000\.0, dtype=np\.float32\),\s*#\s*\[14\][^\n]*\n"
    r"\s*np\.full\(total, sim\[\"rho\"\] / 10000\.0, dtype=np\.float32\),\s*#\s*\[15\][^\n]*\n"
)

new_block = '''_kappa_per_node = np.zeros(total, dtype=np.float32)
            _Cp_per_node    = np.zeros(total, dtype=np.float32)
            _rho_per_node   = np.zeros(total, dtype=np.float32)
            for _rname, _rdata in sim["region_data"].items():
                _o = _rdata["offset"]; _n = _rdata["n_cells"]
                _mat = REGION_MATERIALS.get(_rname, {"kappa": 80.0, "Cp": 450.0, "rho": 7800.0})
                _kappa_per_node[_o:_o+_n] = _mat["kappa"]
                _Cp_per_node[_o:_o+_n]    = _mat["Cp"]
                _rho_per_node[_o:_o+_n]   = _mat["rho"]
            # [13] kappa (per-region, normalised by 100)
            _kappa_per_node / 100.0,
            # [14] Cp    (per-region, normalised by 1000)
            _Cp_per_node / 1000.0,
            # [15] rho   (per-region, normalised by 10000)
            _rho_per_node / 10000.0,
'''

if not old_pat.search(src):
    print("  WARN: could not locate original [13][14][15] block via regex.")
    print("         The file may already be patched, or the format differs.")
    print("         Please paste lines 300-310 of data/dataset_unified.py.")
else:
    src = old_pat.sub(new_block, src, count=1)
    print("  Replaced single-kappa/Cp/rho with per-region lookup.")

p.write_text(src)
PYEOF
echo "[3/5] dataset_unified.py patched."

# ── STEP 4: Quick syntax check ─────────────────────────────
echo "[4/5] Syntax check:"
python3 -c "import ast; ast.parse(open('configs/base_config.py').read()); print('  configs/base_config.py  OK')" \
    || { echo "  SYNTAX ERROR in base_config.py"; exit 1; }
python3 -c "import ast; ast.parse(open('data/dataset_unified.py').read()); print('  data/dataset_unified.py OK')" \
    || { echo "  SYNTAX ERROR in dataset_unified.py"; exit 1; }

# ── STEP 5: Verification printout ──────────────────────────
echo ""
echo "[5/5] Verification:"
echo ""
echo "  REGION_MATERIALS table:"
python3 -c "
import sys; sys.path.insert(0,'.')
from configs.base_config import REGION_MATERIALS
for r, m in REGION_MATERIALS.items():
    print(f'    {r:18s}  k={m[\"kappa\"]:6.2f}  Cp={m[\"Cp\"]:6.1f}  rho={m[\"rho\"]:7.1f}')
"
echo ""
echo "  Per-region block in dataset_unified.py:"
grep -nE "REGION_MATERIALS|_kappa_per_node|_Cp_per_node|_rho_per_node" data/dataset_unified.py | head -12 | sed 's/^/    /'
echo ""
echo "  Backup: $BK"
echo ""
echo "============================================================"
echo "  DONE. Next: sbatch run_sanity_test.sh"
echo "============================================================"
