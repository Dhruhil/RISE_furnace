#!/bin/bash
#SBATCH --job-name=missing_k8
#SBATCH --account=NAISS2026-4-712
#SBATCH --output=/mimer/NOBACKUP/groups/revar/Dataset_k8/logs/missing_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/Dataset_k8/logs/missing_err_%j.log
#SBATCH --time=13:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=45
#SBATCH -C NOGPU

REVAR=/mimer/NOBACKUP/groups/revar
DATASET_CREATION=$REVAR/Dataset_creation
OUTPUT=$REVAR/Dataset_k8
OPENFOAM_SIF=$REVAR/openfoam_2412.sif
PHYSICSNEMO_SIF=$REVAR/physicsnemo_25.06.sif
GMSH_BIN=$REVAR/gmsh-4.13.1-Linux64-sdk/bin/gmsh
BASE_CASE=$REVAR/base_case_that_runs_chnage
MAX_PARALLEL=45

mkdir -p $OUTPUT/logs

echo "=== GENERATE MISSING SAFE CASES ==="
echo "Start: $(date)"

echo ""
echo "========== STEP 1: Create missing cases =========="
cd $DATASET_CREATION

cat > .env << ENVEOF
BASE_CASE=$BASE_CASE
OUTPUT_DIR=$OUTPUT
CONTAINER_BASE_DIR=$OUTPUT
N_LHS_SAMPLES=45
LHS_SEED=42
MAX_PARALLEL_JOBS=$MAX_PARALLEL
ENVEOF

cp configs/parameters_k8.py configs/parameters.py
apptainer exec $PHYSICSNEMO_SIF pip install python-dotenv -q 2>/dev/null || true
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_missing_cases
cp configs/parameters_coarse.py configs/parameters.py

echo ""
echo "========== STEP 2: Mesh + Simulate (PARALLEL) =========="
GEO_NAME="rise_furnace_mid_part_base_case_coarse"
OPENFOAM_ENV="source /usr/lib/openfoam/openfoam2412/etc/bashrc; . /usr/lib/openfoam/openfoam2412/bin/tools/RunFunctions"

run_one_case() {
    local case_dir=$1
    local case_name=$(basename $case_dir)
    local log_file=$OUTPUT/logs/${case_name}.log

    echo "[$(date +%H:%M:%S)] START: $case_name"

    {
        cd "$case_dir"

        echo "  Step A: Gmsh meshing..."
        $GMSH_BIN "${GEO_NAME}.geo" -3 -format msh2 -o "${GEO_NAME}.msh" 2>&1

        echo "  Step B: OpenFOAM meshing..."
        apptainer exec $OPENFOAM_SIF bash -c "
            $OPENFOAM_ENV
            cd $case_dir
            gmshToFoam ${GEO_NAME}.msh > log.gmshToFoam 2>&1
            topoSet > log.topoSet 2>&1 || true
            restore0Dir || true
            splitMeshRegions -cellZones -overwrite > log.splitMeshRegions 2>&1
        "

        echo "  Step C: viewFactorWall fix..."
        if [ -f "$case_dir/constant/inner_box/polyMesh/boundary" ]; then
            sed -i 's/inGroups        1(wall)/inGroups        2(wall viewFactorWall)/g' \
                "$case_dir/constant/inner_box/polyMesh/boundary"
            echo "  viewFactorWall fixed"
        fi

        apptainer exec $OPENFOAM_SIF bash -c "
            $OPENFOAM_ENV
            cd $case_dir
            createBaffles -region rightFluid -overwrite > log.createBaffles 2>&1 || true
        "

        echo "  Step E: Running simulation..."
        apptainer exec $OPENFOAM_SIF bash -c "
            $OPENFOAM_ENV
            cd $case_dir
            viewFactorsGen -region inner_box > log.viewFactorsGen.inner_box 2>&1
            chtMultiRegionFoam > log.chtMultiRegionFoam 2>&1 || true
            foamToVTK -allRegions > log.foamToVTK 2>&1 || true
        "

        if [ -d "$case_dir/VTK" ]; then
            echo "[$(date +%H:%M:%S)] DONE: $case_name"
        else
            echo "[$(date +%H:%M:%S)] FAILED: $case_name"
        fi
    } > "$log_file" 2>&1 &
}

# Only run NEW cases (case046+)
RUNNING=0
for case_dir in $OUTPUT/case0[4-9]*/ $OUTPUT/case[1-9]*/ ; do
    # Skip existing cases (case001-045)
    case_name=$(basename $case_dir)
    case_num=$(echo $case_name | grep -o 'case[0-9]*' | sed 's/case//')
    [ "$case_num" -le 45 ] 2>/dev/null && continue
    
    # Skip if already has VTK
    [ -d "$case_dir/VTK" ] && continue

    run_one_case "$case_dir"
    RUNNING=$((RUNNING + 1))
    if [ $RUNNING -ge $MAX_PARALLEL ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi
done

echo "  Waiting for simulations..."
wait
echo "  All simulations complete"

echo ""
echo "  Results:"
ok=0; fail=0
for case_dir in $OUTPUT/case0[4-9]*/ $OUTPUT/case[1-9]*/ ; do
    case_name=$(basename $case_dir)
    case_num=$(echo $case_name | grep -o 'case[0-9]*' | sed 's/case//')
    [ "$case_num" -le 45 ] 2>/dev/null && continue
    if [ -d "$case_dir/VTK" ]; then
        ok=$((ok + 1))
    else
        echo "    FAIL: $case_name"
        fail=$((fail + 1))
    fi
done
echo "  New cases: OK=$ok, FAIL=$fail"

echo ""
echo "========== STEP 3: Rebuild datasets =========="
cd $DATASET_CREATION
cp configs/parameters_k8.py configs/parameters.py
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_all_regions_dataset || echo "  All-regions dataset FAILED"
cp configs/parameters_coarse.py configs/parameters.py

echo ""
echo "========== STEP 4: Clean + Report =========="
cd $OUTPUT
apptainer exec $PHYSICSNEMO_SIF python3 scripts/clean_dataset.py 2>/dev/null || python3 clean_dataset.py
apptainer exec $PHYSICSNEMO_SIF python3 scripts/generate_report.py 2>/dev/null || python3 generate_report.py

echo ""
echo "========== DONE: $(date) =========="
ls -lh $OUTPUT/*.h5 2>/dev/null
