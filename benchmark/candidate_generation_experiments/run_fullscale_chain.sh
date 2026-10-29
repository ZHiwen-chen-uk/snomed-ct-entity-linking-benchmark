#!/usr/bin/env bash
# Wait for the abbreviation full-scale run to finish, then start the full-scale
# supervised reranker. Sequential, because both load the ~1GB SapBERT matrix and
# the char n-gram index and would otherwise contend for RAM.
#
# Guard: if the abbrev run has not produced its result file within MAX_WAIT
# seconds, start the reranker anyway rather than blocking forever.
cd "$(dirname "$0")" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE

MARKER="stage2_out/ABBREV_FULLSCALE_RESULTS.md"
MAX_WAIT=${MAX_WAIT:-10800}      # 3 hours
waited=0

echo "[chain] waiting for $MARKER ..."
while [ ! -f "$MARKER" ] && [ "$waited" -lt "$MAX_WAIT" ]; do
  sleep 60
  waited=$((waited + 60))
  if [ $((waited % 600)) -eq 0 ]; then
    echo "[chain] still waiting (${waited}s)"
  fi
done

if [ -f "$MARKER" ]; then
  echo "[chain] abbrev finished after ${waited}s -> starting reranker"
else
  echo "[chain] TIMEOUT after ${waited}s, abbrev never produced $MARKER -> starting reranker anyway"
fi

python -u stage2_reranker_full.py
echo "[chain] reranker exited with status $?"
