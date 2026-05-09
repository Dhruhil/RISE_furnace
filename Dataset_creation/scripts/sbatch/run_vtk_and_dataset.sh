#!/bin/bash
#SBATCH --job-name=vtk_dataset
#SBATCH --account=NAISS2026-4-712
#SBATCH --output=/mimer/NOBACKUP/groups/revar/Dataset_k8/logs/vtk_dataset_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/Dataset_k8/logs/vtk_dataset_err_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH -C NOGPU

REVAR=/mimer/NOBACKUP/groups/revar
OUTPUT=$REVAR/Dataset_k8
OPENFOAM_SIF=$REVAR/openfoam_2412.sif
PHYSICSNEMO_SIF=$REVAR/physicsnemo_25.06.sif
DATASET_CREATION=$REVAR/Dataset_creation

echo "=== foamToVTK + Dataset Creation ==="
echo "Start: $(date)"

echo ""
echo "========== foamToVTK (PARALLEL) =========="
RUNNING=0
MAX_PARALLEL=8

for case_dir in $OUTPUT/case*/; do
    case_name=$(basename $case_dir)
    
    if [ -d "$case_dir/VTK" ]; then
        echo "  SKIP (VTK exists): $case_name"
        continue
    fi
    
    if [ ! -f "$case_dir/log.chtMultiRegionFoam" ]; then
        echo "  SKIP (no simulation): $case_name"
        continue
    fi
    
    echo "  foamToVTK: $case_name"
    (
        apptainer exec $OPENFOAM_SIF bash -c "
            source /usr/lib/openfoam/openfoam2412/etc/bashrc
            cd $case_dir
            foamToVTK -allRegions > log.foamToVTK 2>&1 || true
        "
    ) &
    
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done

echo "  Waiting for foamToVTK..."
wait
echo "  foamToVTK complete"

echo ""
echo "  VTK Results:"
ok=0; fail=0
for case_dir in $OUTPUT/case*/; do
    if [ -d "$case_dir/VTK" ]; then
        ok=$((ok + 1))
    else
        echo "    NO VTK: $(basename $case_dir)"
        fail=$((fail + 1))
    fi
done
echo "  VTK OK: $ok, FAIL: $fail"

echo ""
echo "========== Build Datasets =========="
cd $DATASET_CREATION
cp configs/parameters_k8.py configs/parameters.py

cat > .env << ENVEOF
BASE_CASE=$REVAR/base_case_that_runs_chnage
OUTPUT_DIR=$OUTPUT
CONTAINER_BASE_DIR=$OUTPUT
N_LHS_SAMPLES=45
LHS_SEED=42
MAX_PARALLEL_JOBS=45
ENVEOF

apptainer exec $PHYSICSNEMO_SIF pip install python-dotenv -q 2>/dev/null || true
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_dataset || echo "  Steel dataset FAILED"
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_all_regions_dataset || echo "  All-regions dataset FAILED"

cp configs/parameters_coarse.py configs/parameters.py

echo ""
echo "========== DONE: $(date) =========="
ls -lh $OUTPUT/*.h5 2>/dev/null || echo "  No .h5 files created"
