#!/usr/bin/env bash
#SBATCH --job-name=cif-parse-coordinator
#SBATCH -c 1
#SBATCH -t 7-00:00:00
#SBATCH --mem=8g
#SBATCH -p sugon
#SBATCH -o slurm_logs/coordinator-%x-%j.out

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_slurm_multinode.sh --input-list inputs.txt --outdir OUT --shards N [options] -- [cif-parse batch args...]

Coordinator script for multi-node Slurm parsing.  The coordinator itself uses
one CPU.  It splits the input list, submits one multi-core Slurm job per shard,
waits for all shards, then merges shard manifests/reviews/metadata into OUT.

Required:
  --input-list PATH          Text file with one CIF path per line
  --outdir PATH              Final output directory
  --shards N                 Number of shard jobs to submit

Worker resources:
  --jobs-per-shard N         cif-parse --jobs per shard (default: 32)
  --time VALUE               sbatch --time for shard jobs (default: 24:00:00)
  --mem VALUE                sbatch --mem for shard jobs (default: 0; cluster default)
  --partition VALUE          sbatch partition
  --account VALUE            sbatch account
  --qos VALUE                sbatch qos
  --job-name VALUE           Slurm job name prefix (default: cif-parse)
  --sbatch-extra VALUE       Extra sbatch option, repeatable

Execution:
  --config PATH              cif-parse config.toml; passed before the batch subcommand
  --cif-parse-cmd VALUE      Command before "batch" (default: python -m cif_parse.cli)
                             Example: --cif-parse-cmd "mamba run -n bioinfo cif-parse"
  --python VALUE             Python used for split/merge helpers (default: python)
  Parse-stage options such as --input-assembly, --assembly-mode, and
  --max-assembly-atoms must appear after "--" so they are forwarded to
  cif-parse batch on every shard.  --metadata-cif-dir and --metadata-table may
  be passed either to this wrapper or after "--"; paths are normalized before
  shard submission.
  --local-run                Run shard workers locally instead of sbatch
  --local-parallel N         Number of local shard workers (default: 1)
  --wait-interval SEC        Poll interval while waiting for Slurm jobs (default: 30)
  --resume                   Do not resubmit shards with an existing manifest.json.gz
  --merge-only               Only merge existing shard outputs
  --dry-run                  Split and print worker commands without running them

Examples:
  # Submit coordinator from login node; coordinator submits shard jobs.
  ./run_slurm_multinode.sh \
    --input-list test_large.txt \
    --outdir batch_outputs/slurm_parse \
    --shards 8 \
    --jobs-per-shard 64 \
    --config config.toml \
    --partition cpu \
    --time 48:00:00 \
    --cif-parse-cmd "mamba run -n bioinfo cif-parse" \
    -- --assembly-mode all --max-assembly-atoms 300000

  # Submit the coordinator itself as a one-CPU Slurm job.
  sbatch ./run_slurm_multinode.sh --input-list test_large.txt --outdir OUT --shards 8 --jobs-per-shard 64 -- --assembly-mode all

  # Local smoke test without Slurm.
  ./run_slurm_multinode.sh --input-list small.txt --outdir OUT --shards 2 --jobs-per-shard 4 --local-run -- --assembly-mode first_assembly
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

quote_args() {
    local out=""
    local arg
    for arg in "$@"; do
        printf -v out '%s%q ' "$out" "$arg"
    done
    printf '%s' "$out"
}

abs_dir_arg() {
    local path="$1"
    [[ -d "$path" ]] || die "directory not found: $path"
    (cd "$path" && pwd)
}

abs_file_arg() {
    local path="$1"
    [[ -e "$path" ]] || die "file not found: $path"
    printf '%s/%s' "$(cd "$(dirname "$path")" && pwd)" "$(basename "$path")"
}

normalize_parse_path_args() {
    local normalized=()
    local i=0
    local arg value
    while [[ "$i" -lt "${#PARSE_ARGS[@]}" ]]; do
        arg="${PARSE_ARGS[$i]}"
        case "$arg" in
            --metadata-cif-dir)
                value="${PARSE_ARGS[$((i + 1))]:-}"
                [[ -n "$value" ]] || die "--metadata-cif-dir requires a path"
                value="$(abs_dir_arg "$value")"
                METADATA_CIF_DIR="$value"
                normalized+=("$arg" "$value")
                i=$((i + 2))
                ;;
            --metadata-cif-dir=*)
                value="${arg#--metadata-cif-dir=}"
                value="$(abs_dir_arg "$value")"
                METADATA_CIF_DIR="$value"
                normalized+=("--metadata-cif-dir=$value")
                i=$((i + 1))
                ;;
            --metadata-table)
                value="${PARSE_ARGS[$((i + 1))]:-}"
                [[ -n "$value" ]] || die "--metadata-table requires a path"
                value="$(abs_file_arg "$value")"
                METADATA_TABLE="$value"
                normalized+=("$arg" "$value")
                i=$((i + 2))
                ;;
            --metadata-table=*)
                value="${arg#--metadata-table=}"
                value="$(abs_file_arg "$value")"
                METADATA_TABLE="$value"
                normalized+=("--metadata-table=$value")
                i=$((i + 1))
                ;;
            *)
                normalized+=("$arg")
                i=$((i + 1))
                ;;
        esac
    done
    PARSE_ARGS=("${normalized[@]}")
}

INPUT_LIST=""
OUTDIR=""
SHARDS=""
JOBS_PER_SHARD=32
SBATCH_TIME="3-00:00:00"
SBATCH_MEM="250g"
SBATCH_PARTITION="cpu1,cpu2,fat,sugon,hygon"
SBATCH_ACCOUNT=""
SBATCH_QOS=""
JOB_NAME="cif-parse"
CIF_PARSE_CMD="${CIF_PARSE_CMD:-cif-parse}"
CONFIG_PATH=""
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_RUN=0
LOCAL_PARALLEL=1
WAIT_INTERVAL=30
RESUME=0
MERGE_ONLY=0
DRY_RUN=0
SBATCH_EXTRA=()
PARSE_ARGS=()
METADATA_CIF_DIR=""
METADATA_TABLE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --input-list)
            INPUT_LIST="${2:-}"
            shift 2
            ;;
        --outdir)
            OUTDIR="${2:-}"
            shift 2
            ;;
        --shards)
            SHARDS="${2:-}"
            shift 2
            ;;
        --jobs-per-shard)
            JOBS_PER_SHARD="${2:-}"
            shift 2
            ;;
        --time)
            SBATCH_TIME="${2:-}"
            shift 2
            ;;
        --mem)
            SBATCH_MEM="${2:-}"
            shift 2
            ;;
        --partition)
            SBATCH_PARTITION="${2:-}"
            shift 2
            ;;
        --account)
            SBATCH_ACCOUNT="${2:-}"
            shift 2
            ;;
        --qos)
            SBATCH_QOS="${2:-}"
            shift 2
            ;;
        --job-name)
            JOB_NAME="${2:-}"
            shift 2
            ;;
        --cif-parse-cmd)
            CIF_PARSE_CMD="${2:-}"
            shift 2
            ;;
        --config)
            CONFIG_PATH="${2:-}"
            shift 2
            ;;
        --metadata-cif-dir)
            METADATA_CIF_DIR="${2:-}"
            shift 2
            PARSE_ARGS+=("--metadata-cif-dir" "$METADATA_CIF_DIR")
            ;;
        --metadata-table)
            METADATA_TABLE="${2:-}"
            shift 2
            PARSE_ARGS+=("--metadata-table" "$METADATA_TABLE")
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            shift 2
            ;;
        --local-run)
            LOCAL_RUN=1
            shift
            ;;
        --local-parallel)
            LOCAL_PARALLEL="${2:-}"
            shift 2
            ;;
        --wait-interval)
            WAIT_INTERVAL="${2:-}"
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --merge-only)
            MERGE_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --sbatch-extra)
            SBATCH_EXTRA+=("${2:-}")
            shift 2
            ;;
        --)
            shift
            PARSE_ARGS+=("$@")
            break
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$INPUT_LIST" ]] || die "--input-list is required"
[[ -n "$OUTDIR" ]] || die "--outdir is required"
[[ -n "$SHARDS" ]] || die "--shards is required"
[[ "$SHARDS" =~ ^[0-9]+$ ]] && [[ "$SHARDS" -ge 1 ]] || die "--shards must be >= 1"
[[ "$JOBS_PER_SHARD" =~ ^[0-9]+$ ]] && [[ "$JOBS_PER_SHARD" -ge 1 ]] || die "--jobs-per-shard must be >= 1"
[[ "$LOCAL_PARALLEL" =~ ^[0-9]+$ ]] && [[ "$LOCAL_PARALLEL" -ge 1 ]] || die "--local-parallel must be >= 1"
[[ -f "$INPUT_LIST" ]] || die "input list not found: $INPUT_LIST"
if [[ -n "$CONFIG_PATH" ]]; then
    [[ -f "$CONFIG_PATH" ]] || die "config file not found: $CONFIG_PATH"
    CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
fi

if [[ -n "$METADATA_CIF_DIR" ]]; then
    METADATA_CIF_DIR="$(abs_dir_arg "$METADATA_CIF_DIR")"
fi
if [[ -n "$METADATA_TABLE" ]]; then
    METADATA_TABLE="$(abs_file_arg "$METADATA_TABLE")"
fi
normalize_parse_path_args

if [[ "$LOCAL_RUN" -eq 0 && "$MERGE_ONLY" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found; use --local-run for local smoke tests"
    command -v squeue >/dev/null 2>&1 || die "squeue not found; use --local-run for local smoke tests"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$(mkdir -p "$OUTDIR" && cd "$OUTDIR" && pwd)"
INPUT_LIST="$(cd "$(dirname "$INPUT_LIST")" && pwd)/$(basename "$INPUT_LIST")"
WORK_DIR="$OUTDIR/slurm"
SHARD_LIST_DIR="$WORK_DIR/shard_lists"
SHARD_OUT_DIR="$OUTDIR/shards"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$SHARD_LIST_DIR" "$SHARD_OUT_DIR" "$LOG_DIR"

echo "Coordinator: $HOSTNAME"
echo "Input list : $INPUT_LIST"
echo "Output dir : $OUTDIR"
echo "Shards     : $SHARDS"
echo "Jobs/shard : $JOBS_PER_SHARD"
echo "Command    : $CIF_PARSE_CMD"
echo "Config     : ${CONFIG_PATH:-(none)}"
echo "Meta CIF   : ${METADATA_CIF_DIR:-(none)}"
echo "Meta table : ${METADATA_TABLE:-(none)}"
echo "Parse args : ${PARSE_ARGS[*]:-(none)}"

if [[ "$MERGE_ONLY" -eq 0 ]]; then
    "$PYTHON_BIN" "$REPO_ROOT/scripts/split_slurm_shards.py" \
        --input-list "$INPUT_LIST" \
        --shard-dir "$SHARD_LIST_DIR" \
        --shards "$SHARDS"
fi

mapfile -t SHARD_LISTS < <(find "$SHARD_LIST_DIR" -maxdepth 1 -name 'shard_*.txt' -type f | sort)
[[ "${#SHARD_LISTS[@]}" -gt 0 ]] || die "No shard lists found in $SHARD_LIST_DIR"

PARSE_ARGS_QUOTED="$(quote_args "${PARSE_ARGS[@]}")"
CONFIG_ARG_QUOTED=""
if [[ -n "$CONFIG_PATH" ]]; then
    CONFIG_ARG_QUOTED="$(quote_args --config "$CONFIG_PATH")"
fi
WORKER_SCRIPT="$WORK_DIR/run_shard_worker.sh"
cat > "$WORKER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SHARD_ID="\${1:?shard id}"
SHARD_LIST="\${2:?shard list}"
SHARD_OUT="\${3:?shard output dir}"
JOBS="\${4:?jobs}"
mkdir -p "\$SHARD_OUT"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src\${PYTHONPATH:+:\$PYTHONPATH}"
export MPLCONFIGDIR="\${MPLCONFIGDIR:-/tmp/mpl-cif-parse-\${SLURM_JOB_ID:-\$\$}}"
mkdir -p "\$MPLCONFIGDIR"
echo "Shard \$SHARD_ID on \${HOSTNAME}: \$(wc -l < "\$SHARD_LIST") inputs, \$JOBS workers"
set +e
$CIF_PARSE_CMD $CONFIG_ARG_QUOTED batch --input-list "\$SHARD_LIST" --outdir "\$SHARD_OUT" --jobs "\$JOBS" $PARSE_ARGS_QUOTED
status=\$?
set -e
echo "\$status" > "\$SHARD_OUT/.exit_code"
if [[ -f "\$SHARD_OUT/manifest.json.gz" ]]; then
    touch "\$SHARD_OUT/.done"
fi
exit "\$status"
EOF
chmod +x "$WORKER_SCRIPT"

submit_or_run_shards() {
    local shard_list shard_base shard_id shard_out
    JOB_IDS=()
    LOCAL_PIDS=()
    for shard_list in "${SHARD_LISTS[@]}"; do
        shard_base="$(basename "$shard_list" .txt)"
        shard_id="${shard_base#shard_}"
        shard_out="$SHARD_OUT_DIR/$shard_base"
        if [[ "$RESUME" -eq 1 && -f "$shard_out/manifest.json.gz" ]]; then
            echo "Resume: skip existing $shard_base"
            continue
        fi

        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "DRY-RUN: $WORKER_SCRIPT $shard_id $shard_list $shard_out $JOBS_PER_SHARD"
            continue
        fi

        if [[ "$LOCAL_RUN" -eq 1 ]]; then
            while [[ "$(jobs -rp | wc -l)" -ge "$LOCAL_PARALLEL" ]]; do
                sleep 2
            done
            "$WORKER_SCRIPT" "$shard_id" "$shard_list" "$shard_out" "$JOBS_PER_SHARD" \
                > "$LOG_DIR/${shard_base}.out" 2> "$LOG_DIR/${shard_base}.err" &
            LOCAL_PIDS+=("$!")
        else
            sbatch_args=(
                --parsable
                --job-name "${JOB_NAME}_${shard_id}"
                --nodes 1
                --ntasks 1
                --cpus-per-task "$JOBS_PER_SHARD"
                --time "$SBATCH_TIME"
                --output "$LOG_DIR/parse-%x-%j.out"
                --error "$LOG_DIR/parse-%x-%j.err"
            )
            [[ -z "$SBATCH_MEM" ]] || sbatch_args+=(--mem "$SBATCH_MEM")
            [[ -z "$SBATCH_PARTITION" ]] || sbatch_args+=(--partition "$SBATCH_PARTITION")
            [[ -z "$SBATCH_ACCOUNT" ]] || sbatch_args+=(--account "$SBATCH_ACCOUNT")
            [[ -z "$SBATCH_QOS" ]] || sbatch_args+=(--qos "$SBATCH_QOS")
            sbatch_args+=("${SBATCH_EXTRA[@]}")
            jid="$(sbatch "${sbatch_args[@]}" "$WORKER_SCRIPT" "$shard_id" "$shard_list" "$shard_out" "$JOBS_PER_SHARD")"
            jid="${jid%%;*}"
            JOB_IDS+=("$jid")
            echo "Submitted $shard_base as job $jid"
        fi
    done
}

if [[ "$MERGE_ONLY" -eq 0 ]]; then
    submit_or_run_shards
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete; not merging."
    exit 0
fi

if [[ "$MERGE_ONLY" -eq 0 && "$LOCAL_RUN" -eq 1 && "${#LOCAL_PIDS[@]}" -gt 0 ]]; then
    local_status=0
    for pid in "${LOCAL_PIDS[@]}"; do
        if ! wait "$pid"; then
            local_status=1
        fi
    done
    [[ "$local_status" -eq 0 ]] || echo "One or more local shard workers exited non-zero; merging available manifests." >&2
fi

if [[ "$MERGE_ONLY" -eq 0 && "$LOCAL_RUN" -eq 0 && "${#JOB_IDS[@]}" -gt 0 ]]; then
    job_csv="$(IFS=,; echo "${JOB_IDS[*]}")"
    echo "Waiting for Slurm jobs: $job_csv"
    while squeue -h -j "$job_csv" | grep -q .; do
        sleep "$WAIT_INTERVAL"
    done
fi

echo "Merging shard outputs..."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cif-parse-merge-${SLURM_JOB_ID:-$$}}"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON_BIN" "$REPO_ROOT/scripts/merge_slurm_shards.py" \
    --outdir "$OUTDIR" \
    --shard-out-dir "$SHARD_OUT_DIR" \
    --repo-root "$REPO_ROOT"

echo "Done: $OUTDIR"
