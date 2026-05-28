#!/usr/bin/env bash
# =============================================================================
# watch_pipeline.sh — macOS bash 3 compatible
# Monitors downloads, chains index computation, rebuilds MME automatically.
# =============================================================================

YEAR=2026
MONTH=5
DAY=1
INTERVAL=90

while [[ $# -gt 0 ]]; do
  case $1 in
    --year)     YEAR="$2";     shift 2 ;;
    --month)    MONTH="$2";    shift 2 ;;
    --day)      DAY="$2";      shift 2 ;;
    --country)  COUNTRY="$2";  shift 2 ;;
    --root)     ROOT="$2";     shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COUNTRY="${COUNTRY:-${AGROCLIMATE_COUNTRY:-ethiopia}}"
ROOT="${ROOT:-${AGROCLIMATE_CDS_ROOT:-$PROJECT_ROOT/data/countries/$COUNTRY/seasonal/cds}}"
PYTHON="${AGROCLIMATE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export AGROCLIMATE_PROJECT_ROOT="${AGROCLIMATE_PROJECT_ROOT:-$PROJECT_ROOT}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODEL_BASE="$ROOT/seasonal-original-single-levels/$YEAR/$(printf '%02d' "$MONTH")/$(printf '%02d' "$DAY")"
LOG_DIR="$ROOT/logs"
STATE_DIR="$LOG_DIR/.watcher_state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/watcher_$(date '+%Y-%m-%d').log"; }

model_folder() {
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
  esac
}

ALL_MODELS="ecmwf ukmo meteo_france dwd cmcc ncep jma eccc bom"

log "=== watch_pipeline.sh started for $COUNTRY (checking every ${INTERVAL}s) ==="

while true; do
  NEWLY_INDEXED=0
  N_READY=0

  for MODEL in $ALL_MODELS; do
    FOLDER="$(model_folder "$MODEL")"
    MODEL_DIR="$MODEL_BASE/$FOLDER"
    IDX_FLAG="$MODEL_DIR/indices/ensemble_statistics.nc"
    COMPUTE_STARTED_FLAG="$STATE_DIR/${MODEL}_compute_started"
    IDX_NOTIFIED_FLAG="$STATE_DIR/${MODEL}_idx_notified"

    if [[ -f "$IDX_FLAG" ]]; then
      N_READY=$((N_READY + 1))
      if [[ ! -f "$IDX_NOTIFIED_FLAG" ]]; then
        log "  ✓ Indices ready: $MODEL"
        touch "$IDX_NOTIFIED_FLAG"
        NEWLY_INDEXED=1
      fi
      continue
    fi

    # Count raw files
    RAW_COUNT=$(find "$MODEL_DIR" -maxdepth 1 -name '*.nc' 2>/dev/null | wc -l | tr -d ' ')
    DERIVED_COUNT=$(find "$MODEL_DIR/derived" -name '*.nc' 2>/dev/null | wc -l | tr -d ' ')

    if [[ "$RAW_COUNT" -ge 9 && "$DERIVED_COUNT" -ge 3 && ! -f "$COMPUTE_STARTED_FLAG" ]]; then
      DATE_STR=$(printf "%04d-%02d-%02d" "$YEAR" "$MONTH" "$DAY")
      IDX_LOG="$LOG_DIR/indices_${MODEL}_${DATE_STR}.log"
      log "  → Data ready: $MODEL ($RAW_COUNT raw, $DERIVED_COUNT derived) — computing indices …"
      touch "$COMPUTE_STARTED_FLAG"
      (
        "$PYTHON" -m cds_agroclimate_pipeline.cds.compute_indices \
            --year "$YEAR" --month "$MONTH" --day "$DAY" \
            --country "$COUNTRY" --root "$ROOT" --coarsen 1 \
            --models "$MODEL" --workers 8 \
            >> "$IDX_LOG" 2>&1
        log "  ✓ Index computation done: $MODEL"
      ) &
    elif [[ "$RAW_COUNT" -gt 0 && "$RAW_COUNT" -lt 9 ]]; then
      log "  ⏳ Downloading: $MODEL (${RAW_COUNT}/9 files)"
    fi
  done

  # Rebuild MME + brief when new indices arrive
  if [[ "$NEWLY_INDEXED" -eq 1 && "$N_READY" -ge 2 ]]; then
    DATE_STR=$(printf "%04d-%02d-%02d" "$YEAR" "$MONTH" "$DAY")
    log "  → Rebuilding MME with $N_READY models …"
    "$PYTHON" -m cds_agroclimate_pipeline.cds.build_mme \
        --year "$YEAR" --month "$MONTH" --day "$DAY" \
        --country "$COUNTRY" --root "$ROOT" --min-models 2 \
        >> "$LOG_DIR/mme_${DATE_STR}.log" 2>&1
    log "  ✓ MME rebuilt ($N_READY models)"

    log "  → Regenerating MME brief …"
    "$PYTHON" -m cds_agroclimate_pipeline.publication.mme_brief \
        --year "$YEAR" --month "$MONTH" --day "$DAY" \
        --country "$COUNTRY" --root "$ROOT" \
        >> "$LOG_DIR/mme_brief_${DATE_STR}.log" 2>&1
    log "  ✓ MME brief regenerated"
  fi

  # Status
  PENDING=""
  for MODEL in $ALL_MODELS; do
    FOLDER="$(model_folder "$MODEL")"
    [[ ! -f "$MODEL_BASE/$FOLDER/indices/ensemble_statistics.nc" ]] && PENDING="$PENDING $MODEL"
  done
  PENDING="${PENDING# }"

  if [[ -z "$PENDING" ]]; then
    log "  🎉 ALL 9 MODELS COMPLETE! Final MME brief ready."
    log "  → $MODEL_BASE/multimodel_ensemble/publication/MoA_MME_Advisory_Kiremt${YEAR}.pdf"
    break
  fi

  log "  Status: $N_READY/9 indexed  |  pending:$PENDING"
  sleep "$INTERVAL"
done
