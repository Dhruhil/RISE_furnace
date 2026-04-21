#!/bin/bash
set -u
cd /mimer/NOBACKUP/groups/revar/GNN_Unified || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
BK=backup_v2fix_${STAMP}
mkdir -p "$BK"
cp data/dataset_unified.py "$BK/dataset_unified.py.bak"
echo "Backup: $BK/dataset_unified.py.bak"

python3 << 'PYEOF'
import pathlib, re

p = pathlib.Path("data/dataset_unified.py")
src = p.read_text()

# ── FIX 1: remove the misplaced import inside the method ────
# Pattern: "        import random\nfrom configs.base_config import REGION_MATERIALS\n"
bad = "        import random\nfrom configs.base_config import REGION_MATERIALS\n"
good = "        import random\n"
if bad in src:
    src = src.replace(bad, good, 1)
    print("[1] Removed misplaced REGION_MATERIALS import from inside method.")
else:
    print("[1] Misplaced import not found (may already be fixed).")

# Ensure top-level import exists (add after the first block of imports)
if "from configs.base_config import REGION_MATERIALS" not in src:
    # Find the first blank line after the opening import block and insert before it
    m = re.search(r"^(import [^\n]+\n|from [^\n]+\n)+", src, flags=re.MULTILINE)
    if m:
        end = m.end()
        src = src[:end] + "from configs.base_config import REGION_MATERIALS\n" + src[end:]
        print("[2] Added REGION_MATERIALS import at top of file.")
    else:
        print("[2] ERROR: could not find import block at top of file.")
else:
    print("[2] REGION_MATERIALS import already at top (or will be after fix 1).")

# ── FIX 3: move per-region assignments OUT of np.column_stack list ──
# Locate the broken block and replace with clean version.
broken_pat = re.compile(
    r"            np\.full\(total, sim\[\"height\"\] / 0\.20, dtype=np\.float32\),\s*#\s*\[12\][^\n]*\n"
    r"            _kappa_per_node = np\.zeros\(total, dtype=np\.float32\)\n"
    r"            _Cp_per_node    = np\.zeros\(total, dtype=np\.float32\)\n"
    r"            _rho_per_node   = np\.zeros\(total, dtype=np\.float32\)\n"
    r"            for _rname, _rdata in sim\[\"region_data\"\]\.items\(\):\n"
    r"                _o = _rdata\[\"offset\"\]; _n = _rdata\[\"n_cells\"\]\n"
    r"                _mat = REGION_MATERIALS\.get\(_rname, \{\"kappa\": 80\.0, \"Cp\": 450\.0, \"rho\": 7800\.0\}\)\n"
    r"                _kappa_per_node\[_o:_o\+_n\] = _mat\[\"kappa\"\]\n"
    r"                _Cp_per_node\[_o:_o\+_n\]    = _mat\[\"Cp\"\]\n"
    r"                _rho_per_node\[_o:_o\+_n\]   = _mat\[\"rho\"\]\n"
    r"            # \[13\] kappa [^\n]*\n"
    r"            _kappa_per_node / 100\.0,\n"
    r"            # \[14\] Cp [^\n]*\n"
    r"            _Cp_per_node / 1000\.0,\n"
    r"            # \[15\] rho [^\n]*\n"
    r"            _rho_per_node / 10000\.0,\n"
)

if not broken_pat.search(src):
    print("[3] Broken block NOT found by regex.")
    print("    Dumping what I see at the target location for debugging:")
    # Find surroundings
    m2 = re.search(r'#\s*\[12\][^\n]*\n[^\n]*\n[^\n]*\n', src)
    if m2:
        print(src[max(0,m2.start()-100):m2.end()+400])
    raise SystemExit(1)

replacement = (
    '            np.full(total, sim["height"] / 0.20, dtype=np.float32),    # [12] height\n'
    '            _kappa_feat,    # [13] kappa  (per-region, /100)\n'
    '            _Cp_feat,       # [14] Cp     (per-region, /1000)\n'
    '            _rho_feat,      # [15] rho    (per-region, /10000)\n'
)
src = broken_pat.sub(replacement, src, count=1)

# Now inject the 3 arrays BEFORE np.column_stack([
# We need to locate the "node_feats = np.column_stack([" line in __getitem__
# and insert the per-region computation right above it.
injection = (
    "        # Per-region material properties (from REGION_MATERIALS)\n"
    "        _kappa_feat = np.zeros(total, dtype=np.float32)\n"
    "        _Cp_feat    = np.zeros(total, dtype=np.float32)\n"
    "        _rho_feat   = np.zeros(total, dtype=np.float32)\n"
    "        for _rname, _rdata in sim[\"region_data\"].items():\n"
    "            _o = _rdata[\"offset\"]; _n = _rdata[\"n_cells\"]\n"
    "            _mat = REGION_MATERIALS.get(_rname, {\"kappa\": 80.0, \"Cp\": 450.0, \"rho\": 7800.0})\n"
    "            _kappa_feat[_o:_o+_n] = _mat[\"kappa\"] / 100.0\n"
    "            _Cp_feat[_o:_o+_n]    = _mat[\"Cp\"] / 1000.0\n"
    "            _rho_feat[_o:_o+_n]   = _mat[\"rho\"] / 10000.0\n"
    "\n"
)

# Find "node_feats = np.column_stack([" with some leading whitespace
m3 = re.search(r"^(\s*)node_feats\s*=\s*np\.column_stack\(\[", src, flags=re.MULTILINE)
if m3 is None:
    # Maybe the variable has a different name — find column_stack near our replacement
    m3 = re.search(r"^(\s*)[a-zA-Z_]+\s*=\s*np\.column_stack\(\[", src, flags=re.MULTILINE)
if m3 is None:
    print("[3] ERROR: cannot find np.column_stack([ assignment.")
    raise SystemExit(1)

indent = m3.group(1)
# Re-indent injection to match
inj_indented = "\n".join(
    (indent + line[8:]) if line.startswith("        ") else (indent + line if line.strip() else line)
    for line in injection.splitlines()
) + "\n"

src = src[:m3.start()] + inj_indented + src[m3.start():]
print("[3] Restructured per-region block (now outside np.column_stack).")

p.write_text(src)

# Syntax check
import ast
try:
    ast.parse(src)
    print("[4] Syntax check: OK")
except SyntaxError as e:
    print(f"[4] Syntax check: FAILED — {e}")
    raise SystemExit(1)
PYEOF

echo ""
echo "=== VERIFICATION ==="
echo ""
echo "--- Imports at top of file ---"
head -20 data/dataset_unified.py | grep -nE "import|from"
echo ""
echo "--- Lines around split (105-120) ---"
sed -n '105,120p' data/dataset_unified.py
echo ""
echo "--- Per-region block (look for _kappa_feat) ---"
grep -n "_kappa_feat\|_Cp_feat\|_rho_feat\|REGION_MATERIALS" data/dataset_unified.py | head -20
echo ""
echo "=== DONE ==="
