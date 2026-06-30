#!/usr/bin/env bash
# Attractor test (decisive metric 5): at chosen cells, perturb theta_LR BOTH ways and
# check whether it returns to theta*. egate sustained cells should return from both
# sides; cos's interior crossing should NOT (fuses or segregates); diff is a fusion ref.
set -u
cd "$(dirname "$0")"
SIG=${SIG:-0.016}
SEEDS="0 1 2 3 4 5"
# (kind delta mu)
CELLS=(
  "egate 0.1 0.6" "egate 0.1 0.9" "egate 0.05 0.6"
  "cos 0.1 0.9"   "cos 0.1 0.6"
  "diff 0.1 0.9"
)
{
  for cell in "${CELLS[@]}"; do
    set -- $cell
    for s in $SEEDS; do
      echo "python3 bifurcate.py attractor $1 $2 $3 --sigma $SIG --gate_phase --seed $s"
    done
  done
} | xargs -P 4 -I CMD bash -c CMD
echo "=== ATTRACTOR TESTS DONE ==="
