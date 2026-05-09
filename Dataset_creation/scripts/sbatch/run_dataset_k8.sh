#!/bin/bash
#SBATCH --job-name=dataset_k8
#SBATCH --account=NAISS2026-4-712
#SBATCH --output=/mimer/NOBACKUP/groups/revar/Dataset_k8/logs/pipeline_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/Dataset_k8/logs/pipeline_err_%j.log
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=45
#SBATCH -C NOGPU

# set -e  # disabled: simulations may crash

REVAR=/mimer/NOBACKUP/groups/revar
DATASET_CREATION=$REVAR/Dataset_creation
OUTPUT=$REVAR/Dataset_k8
OPENFOAM_SIF=$REVAR/openfoam_2412.sif
PHYSICSNEMO_SIF=$REVAR/physicsnemo_25.06.sif
GMSH_BIN=$REVAR/gmsh-4.13.1-Linux64-sdk/bin/gmsh
BASE_CASE=$REVAR/base_case_that_runs_chnage
MAX_PARALLEL=45

mkdir -p $OUTPUT/logs

echo "=== TEST PIPELINE — 2 Cases PARALLEL ==="
echo "  Start: $(date)"

rm -rf $OUTPUT/case*
rm -rf $OUTPUT/run_all_openfoam.sh
rm -rf $OUTPUT/case_manifest.json
rm -rf $OUTPUT/*.h5

echo ""
echo "========== STEP 1: Generate cases =========="
cd $DATASET_CREATION

cat > .env << ENVEOF
BASE_CASE=$BASE_CASE
OUTPUT_DIR=$OUTPUT
CONTAINER_BASE_DIR=$OUTPUT
N_LHS_SAMPLES=45
LHS_SEED=42
MAX_PARALLEL_JOBS=$MAX_PARALLEL
ENVEOF

apptainer exec $PHYSICSNEMO_SIF pip install python-dotenv -q 2>/dev/null || true
cp configs/parameters_k8.py configs/parameters.py
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_cases
echo "  Cases created"
cp configs/parameters_coarse.py configs/parameters.py

echo ""
echo "========== STEP 2: Validate =========="
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.validate_cases

echo ""
echo "========== STEP 3: Mesh + Simulate (PARALLEL) =========="
GEO_NAME="rise_furnace_mid_part_base_case_coarse"
OPENFOAM_ENV="source /usr/lib/openfoam/openfoam2412/etc/bashrc; . /usr/lib/openfoam/openfoam2412/bin/tools/RunFunctions"

run_one_case() {
    local case_dir=$1
    local case_name=$(basename $case_dir)
    local log_file=$OUTPUT/logs/${case_name}.log

    echo "[$(date +%H:%M:%S)] START: $case_name"

    {
        cd "$case_dir"

        # Step A: Gmsh (external binary)
        echo "  Step A: Gmsh meshing..."
        $GMSH_BIN "${GEO_NAME}.geo" -3 -format msh2 -o "${GEO_NAME}.msh" 2>&1

        # Step B: OpenFOAM meshing (gmshToFoam + splitMeshRegions)
        echo "  Step B: OpenFOAM meshing..."
        apptainer exec $OPENFOAM_SIF bash -c "
            $OPENFOAM_ENV
            cd $case_dir
            rm -f log.gmshToFoam log.topoSet log.splitMeshRegions log.changeDictionary.* log.createBaffles log.viewFactorsGen.* log.chtMultiRegionFoam log.Allrun
            gmshToFoam ${GEO_NAME}.msh > log.gmshToFoam 2>&1
            topoSet > log.topoSet 2>&1 || true
            restore0Dir || true
            splitMeshRegions -cellZones -overwrite > log.splitMeshRegions 2>&1
        "

        # Step C: Fix viewFactorWall (outside container — just sed)
        echo "  Step C: viewFactorWall fix..."
        if [ -f "$case_dir/constant/inner_box/polyMesh/boundary" ]; then
            sed -i 's/inGroups        1(wall)/inGroups        2(wall viewFactorWall)/g' \
                "$case_dir/constant/inner_box/polyMesh/boundary"
            echo "  viewFactorWall fixed"
        fi

        # Step D: Create baffles for radiation
        apptainer exec $OPENFOAM_SIF bash -c "
            $OPENFOAM_ENV
            cd $case_dir
            createBaffles -region rightFluid -overwrite > log.createBaffles 2>&1 || true
        "

        # Step E: Run simulation
        echo "  Step E: Running simulation..."
        apptainer exec $OPENFOAM_SIF bash -c "
            $OPENFOAM_ENV
            cd $case_dir
            rm -f log.viewFactorsGen.inner_box log.chtMultiRegionFoam log.Allrun log.foamToVTK
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

RUNNING=0
for case_dir in $OUTPUT/case*/; do
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
for case_dir in $OUTPUT/case*/; do
    case_name=$(basename $case_dir)
    if [ -d "$case_dir/VTK" ]; then
        echo "    OK: $case_name"
    else
        echo "    FAIL: $case_name"
    fi
done

echo ""
echo "========== STEP 4: Build datasets =========="
cd $DATASET_CREATION
cp configs/parameters_k8.py configs/parameters.py
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_dataset || echo "  Steel dataset failed"
apptainer exec $PHYSICSNEMO_SIF python3 -m scripts.create_all_regions_dataset || echo "  All-regions dataset failed"

cp configs/parameters_coarse.py configs/parameters.py
echo ""
echo "========== DONE: $(date) =========="
ls $OUTPUT/*.h5 2>/dev/null || echo "  No datasets"
