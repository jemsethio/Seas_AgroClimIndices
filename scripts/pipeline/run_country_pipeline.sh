#!/usr/bin/env bash
# =============================================================================
# run_country_pipeline.sh
# =============================================================================
# Complete multi-model pipeline for Ethiopia Kiremt agroclimate advisory:
#
#   Phase 1 — Download (parallel CDS requests for all 9 models)
#   Phase 2 — Compute indices (sequential, 1 model at a time, OOM-safe)
#   Phase 3 — Per-model publication figures + woreda advisory PDF
#   Phase 4 — Multi-Model Ensemble (MME) statistics
#   Phase 5 — MME policy brief PDF (consensus + spread + woreda advisory)
#
# Usage:
#   bash scripts/pipeline/run_country_pipeline.sh --year 2026 --month 5
#   bash scripts/pipeline/run_country_pipeline.sh --year 2026 --month 5 --skip-download
#   bash scripts/pipeline/run_country_pipeline.sh --year 2026 --month 5 --skip-download --skip-compute
#   bash scripts/pipeline/run_country_pipeline.sh --year 2026 --month 5 --models "ecmwf ukmo ncep"
#   bash scripts/pipeline/run_country_pipeline.sh --year 2026 --month 5 --min-models 3
# =============================================================================

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
YEAR=2026
MONTH=5
DAY=1
MODELS_ARG=""
SKIP_DOWNLOAD=false
SKIP_COMPUTE=false
SKIP_PERMODEL=false
SKIP_MME=false
DOWNLOAD_WORKERS=4
COMPUTE_WORKERS=8
MIN_MODELS=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COUNTRY="${AGROCLIMATE_COUNTRY:-ethiopia}"
ROOT="${AGROCLIMATE_CDS_ROOT:-$PROJECT_ROOT/data/countries/$COUNTRY/seasonal/cds}"
PYTHON="${AGROCLIMATE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export AGROCLIMATE_PROJECT_ROOT="${AGROCLIMATE_PROJECT_ROOT:-$PROJECT_ROOT}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
LOG_DIR="$ROOT/logs"

ALL_MODELS=(ecmwf ukmo meteo_france dwd cmcc ncep jma eccc bom)

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --year)              YEAR="$2";             shift 2 ;;
    --month)             MONTH="$2";            shift 2 ;;
    --day)               DAY="$2";              shift 2 ;;
    --country)           COUNTRY="$2"; ROOT="$PROJECT_ROOT/data/countries/$COUNTRY/seasonal/cds"; shift 2 ;;
    --root)              ROOT="$2";             shift 2 ;;
    --models)            MODELS_ARG="$2";       shift 2 ;;
    --skip-download)     SKIP_DOWNLOAD=true;    shift   ;;
    --skip-compute)      SKIP_COMPUTE=true;     shift   ;;
    --skip-permodel)     SKIP_PERMODEL=true;    shift   ;;
    --skip-mme)          SKIP_MME=true;         shift   ;;
    --download-workers)  DOWNLOAD_WORKERS="$2"; shift 2 ;;
    --compute-workers)   COMPUTE_WORKERS="$2";  shift 2 ;;
    --min-models)        MIN_MODELS="$2";       shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
DATE_STR=$(printf "%04d-%02d-%02d" "$YEAR" "$MONTH" "$DAY")
PIPE_LOG="$LOG_DIR/full_pipeline_${DATE_STR}.log"
MODEL_BASE="$ROOT/seasonal-original-single-levels/$YEAR/$(printf '%02d' "$MONTH")/$(printf '%02d' "$DAY")"

if [[ -n "$MODELS_ARG" ]]; then
  read -ra MODEL_LIST <<< "$MODELS_ARG"
else
  MODEL_LIST=("${ALL_MODELS[@]}")
fi

# ── helper functions ──────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PIPE_LOG"; }

model_folder() {
  # ecmwf → ecmwf_system51, ukmo → ukmo_system610, etc.
  case "$1" in
    ecmwf)        echo "ecmwf_system51" ;;
    ukmo)         echo "ukmo_system610" ;;
    meteo_france) echo "meteo_france_system9" ;;
    dwd)          echo "dwd_system22" ;;
    cmcc)         echo "cmcc_system4" ;;
    ncep)         echo "ncep_system2" ;;
    jma)          echo "jma_system4" ;;
    eccc)         echo "eccc_system5" ;;
    bom)          echo "bom_system2" ;;
    *) echo "${1}_systemX" ;;
  esac
}

# ── banner ────────────────────────────────────────────────────────────────────
log ""
log "======================================================================="
log "  Agroclimate Pipeline — Country: $COUNTRY — ALL 9 MODELS + MME"
log "  Date: $DATE_STR  |  Root: $ROOT"
log "  Models: ${MODEL_LIST[*]}"
log "  Phases: download=$([[ $SKIP_DOWNLOAD == false ]] && echo ON || echo SKIP)"
log "          compute=$([[ $SKIP_COMPUTE == false ]] && echo ON || echo SKIP)"
log "          permodel=$([[ $SKIP_PERMODEL == false ]] && echo ON || echo SKIP)"
log "          mme=$([[ $SKIP_MME == false ]] && echo ON || echo SKIP)"
log "======================================================================="


# =============================================================================
# PHASE 1 — DOWNLOAD (parallel CDS requests)
# =============================================================================
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
  log ""
  log "━━━ PHASE 1: DOWNLOAD (${#MODEL_LIST[@]} models in parallel) ━━━"

  dl_pids=()
  dl_models=()

  for MODEL in "${MODEL_LIST[@]}"; do
    FOLDER="$(model_folder "$MODEL")"
    NC_CHECK=$(find "$MODEL_BASE/$FOLDER" -maxdepth 1 -name "total_precipitation.nc" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$NC_CHECK" -gt 0 ]]; then
      log "  SKIP download (already on disk): $MODEL"
      continue
    fi
    DL_LOG="$LOG_DIR/download_${MODEL}_${DATE_STR}.log"
    log "  → starting download: $MODEL"
    (
      "$PYTHON" -m cds_agroclimate_pipeline.cds.download_surface \
          --year "$YEAR" --month "$MONTH" --day "$DAY" \
          --country "$COUNTRY" --root "$ROOT" --workers "$DOWNLOAD_WORKERS" \
          --models "$MODEL" \
          >> "$DL_LOG" 2>&1
    ) &
    dl_pids+=($!)
    dl_models+=("$MODEL")
  done

  dl_failed=()
  for i in "${!dl_pids[@]}"; do
    PID="${dl_pids[$i]}"
    M="${dl_models[$i]}"
    if wait "$PID"; then
      log "  ✓ DOWNLOAD done : $M"
    else
      log "  ✗ DOWNLOAD FAIL : $M  (exit $?) — see $LOG_DIR/download_${M}_${DATE_STR}.log"
      dl_failed+=("$M")
    fi
  done

  [[ ${#dl_failed[@]} -gt 0 ]] && log "  WARNING: ${#dl_failed[@]} download(s) failed: ${dl_failed[*]}"
  log "━━━ PHASE 1 complete ━━━"
else
  log "━━━ PHASE 1: DOWNLOAD skipped ━━━"
fi


# =============================================================================
# PHASE 2 — COMPUTE INDICES (sequential, one model at a time)
# =============================================================================
compute_failed=()

if [[ "$SKIP_COMPUTE" == "false" ]]; then
  log ""
  log "━━━ PHASE 2: COMPUTE INDICES (sequential) ━━━"

  for MODEL in "${MODEL_LIST[@]}"; do
    FOLDER="$(model_folder "$MODEL")"
    DONE_FLAG="$MODEL_BASE/$FOLDER/indices/ensemble_statistics.nc"
    IDX_LOG="$LOG_DIR/indices_${MODEL}_${DATE_STR}.log"

    if [[ -f "$DONE_FLAG" ]]; then
      log "  SKIP compute (already done): $MODEL"
      continue
    fi

    RAW_COUNT=$(find "$MODEL_BASE/$FOLDER" -maxdepth 1 -name '*.nc' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$RAW_COUNT" -eq 0 ]]; then
      log "  SKIP compute (no raw data): $MODEL"
      continue
    fi

    log "  ► COMPUTE start : $MODEL  ($RAW_COUNT raw files)"
    T0=$SECONDS
    "$PYTHON" -m cds_agroclimate_pipeline.cds.compute_indices \
        --year "$YEAR" --month "$MONTH" --day "$DAY" \
        --country "$COUNTRY" --root "$ROOT" --coarsen 1 \
        --models "$MODEL" --workers "$COMPUTE_WORKERS" \
        >> "$IDX_LOG" 2>&1
    EXIT_CODE=$?
    ELAPSED=$(( SECONDS - T0 ))
    if [[ $EXIT_CODE -eq 0 ]]; then
      log "  ✓ COMPUTE done  : $MODEL  (${ELAPSED}s)"
    else
      log "  ✗ COMPUTE FAIL  : $MODEL  (${ELAPSED}s) — see $IDX_LOG"
      compute_failed+=("$MODEL")
    fi
  done
  log "━━━ PHASE 2 complete ━━━"
else
  log "━━━ PHASE 2: COMPUTE skipped ━━━"
fi


# =============================================================================
# PHASE 3 — PER-MODEL: publication figures + woreda advisory PDF
# =============================================================================
permodel_failed=()

if [[ "$SKIP_PERMODEL" == "false" ]]; then
  log ""
  log "━━━ PHASE 3: PER-MODEL OUTPUTS (plots + advisory PDF) ━━━"

  for MODEL in "${MODEL_LIST[@]}"; do
    FOLDER="$(model_folder "$MODEL")"
    DONE_FLAG="$MODEL_BASE/$FOLDER/indices/ensemble_statistics.nc"
    PP_LOG="$LOG_DIR/postproc_${MODEL}_${DATE_STR}.log"

    if [[ ! -f "$DONE_FLAG" ]]; then
      log "  SKIP post-proc (no indices): $MODEL"
      continue
    fi

    PUB_DIR="$MODEL_BASE/$FOLDER/indices/plots/publication"
    BRIEF_FLAG="$PUB_DIR/MoA_ElNino_Advisory_Kiremt${YEAR}_Woreda.pdf"

    if [[ -f "$BRIEF_FLAG" ]]; then
      log "  SKIP post-proc (outputs exist): $MODEL"
      continue
    fi

    log "  ► POST-PROC start : $MODEL"
    T0=$SECONDS

    (
      # 3a. Publication figures
      "$PYTHON" -m cds_agroclimate_pipeline.publication.figures \
          --year "$YEAR" --month "$MONTH" --day "$DAY" \
          --country "$COUNTRY" --root "$ROOT" \
          --model "$MODEL" >> "$PP_LOG" 2>&1

      # 3b. Woreda advisory PDF
      "$PYTHON" -m cds_agroclimate_pipeline.publication.moa_woreda_brief \
          --year "$YEAR" --month "$MONTH" --day "$DAY" \
          --country "$COUNTRY" --root "$ROOT" \
          --model "$MODEL" >> "$PP_LOG" 2>&1
    )

    EXIT_CODE=$?
    ELAPSED=$(( SECONDS - T0 ))
    if [[ $EXIT_CODE -eq 0 ]]; then
      log "  ✓ POST-PROC done  : $MODEL  (${ELAPSED}s)"
    else
      log "  ✗ POST-PROC FAIL  : $MODEL  (exit $EXIT_CODE) — see $PP_LOG"
      permodel_failed+=("$MODEL")
    fi
  done
  log "━━━ PHASE 3 complete ━━━"
else
  log "━━━ PHASE 3: PER-MODEL outputs skipped ━━━"
fi


# =============================================================================
# PHASE 4 — MULTI-MODEL ENSEMBLE (MME) statistics
# =============================================================================
if [[ "$SKIP_MME" == "false" ]]; then
  log ""
  log "━━━ PHASE 4: MULTI-MODEL ENSEMBLE (MME) statistics ━━━"

  MME_PATH="$MODEL_BASE/multimodel_ensemble/mme_statistics.nc"
  MME_LOG="$LOG_DIR/mme_${DATE_STR}.log"

  # Count ready models
  N_READY=0
  for MODEL in "${MODEL_LIST[@]}"; do
    FOLDER="$(model_folder "$MODEL")"
    [[ -f "$MODEL_BASE/$FOLDER/indices/ensemble_statistics.nc" ]] && N_READY=$((N_READY + 1))
  done
  log "  Ready models: $N_READY (min required: $MIN_MODELS)"

  if [[ $N_READY -lt $MIN_MODELS ]]; then
    log "  SKIP MME: not enough models ready ($N_READY < $MIN_MODELS)"
  else
    log "  ► MME computation …"
    T0=$SECONDS
    "$PYTHON" -m cds_agroclimate_pipeline.cds.build_mme \
        --year "$YEAR" --month "$MONTH" --day "$DAY" \
        --country "$COUNTRY" --root "$ROOT" --min-models "$MIN_MODELS" \
        >> "$MME_LOG" 2>&1
    EXIT_CODE=$?
    ELAPSED=$(( SECONDS - T0 ))
    if [[ $EXIT_CODE -eq 0 ]]; then
      log "  ✓ MME done  (${ELAPSED}s)"
    else
      log "  ✗ MME FAIL  (${ELAPSED}s) — see $MME_LOG"
    fi
  fi
  log "━━━ PHASE 4 complete ━━━"
fi


# =============================================================================
# PHASE 5 — MME POLICY BRIEF PDF
# =============================================================================
if [[ "$SKIP_MME" == "false" ]]; then
  log ""
  log "━━━ PHASE 5: MME POLICY BRIEF PDF ━━━"

  MME_PATH="$MODEL_BASE/multimodel_ensemble/mme_statistics.nc"
  BRIEF_LOG="$LOG_DIR/mme_brief_${DATE_STR}.log"

  if [[ ! -f "$MME_PATH" ]]; then
    log "  SKIP MME brief: no mme_statistics.nc"
  else
    log "  ► Generating MME policy brief …"
    T0=$SECONDS
    "$PYTHON" -m cds_agroclimate_pipeline.publication.mme_brief \
        --year "$YEAR" --month "$MONTH" --day "$DAY" \
        --country "$COUNTRY" --root "$ROOT" \
        >> "$BRIEF_LOG" 2>&1
    EXIT_CODE=$?
    ELAPSED=$(( SECONDS - T0 ))
    if [[ $EXIT_CODE -eq 0 ]]; then
      BRIEF_PATH="$MODEL_BASE/multimodel_ensemble/publication/MoA_MME_Advisory_Kiremt${YEAR}.pdf"
      log "  ✓ MME brief done  (${ELAPSED}s)"
      [[ -f "$BRIEF_PATH" ]] && log "  → $BRIEF_PATH"
    else
      log "  ✗ MME brief FAIL  (${ELAPSED}s) — see $BRIEF_LOG"
    fi
  fi
  log "━━━ PHASE 5 complete ━━━"
fi


# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "======================================================================="
log "  PIPELINE SUMMARY — $DATE_STR"
log "======================================================================="
for MODEL in "${MODEL_LIST[@]}"; do
  FOLDER="$(model_folder "$MODEL")"
  NC_RAW=$(find "$MODEL_BASE/$FOLDER" -maxdepth 1 -name '*.nc' 2>/dev/null | wc -l | tr -d ' ')
  HAS_IDX=$([[ -f "$MODEL_BASE/$FOLDER/indices/ensemble_statistics.nc" ]] && echo "✓" || echo "✗")
  HAS_PDF=$([[ -f "$MODEL_BASE/$FOLDER/indices/plots/publication/MoA_ElNino_Advisory_Kiremt${YEAR}_Woreda.pdf" ]] && echo "✓" || echo "✗")
  log "  $MODEL  | raw:$NC_RAW  | indices:$HAS_IDX  | PDF:$HAS_PDF"
done

MME_NC=$([[ -f "$MODEL_BASE/multimodel_ensemble/mme_statistics.nc" ]] && echo "✓" || echo "✗")
MME_PDF=$([[ -f "$MODEL_BASE/multimodel_ensemble/publication/MoA_MME_Advisory_Kiremt${YEAR}.pdf" ]] && echo "✓" || echo "✗")
log "  MME      | mme_statistics:$MME_NC  | MME brief:$MME_PDF"
log ""
[[ ${#compute_failed[@]} -gt 0 ]]  && log "  ✗ Compute failures:  ${compute_failed[*]}"
[[ ${#permodel_failed[@]} -gt 0 ]] && log "  ✗ Post-proc failures: ${permodel_failed[*]}"
log "  Log: $PIPE_LOG"
log "======================================================================="
