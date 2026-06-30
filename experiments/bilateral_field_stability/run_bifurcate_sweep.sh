#!/usr/bin/env bash
# Full delta/mu regime map (gate_phase=True: the coupling law gates BOTH membranes).
# Primary contrast egate vs cos (full grid); diff + iso as references. 4-way parallel,
# per-cell JSON (resumable). N=16, T=2500.
set -u
cd "$(dirname "$0")"
N=${N:-16}; T=${T:-2500}; SIG=${SIG:-0.016}
DELTAS="0.05 0.1 0.2 0.4"
MUS="0.4 0.6 0.9 1.3 1.8"

emit() {  # emit all cell commands
  for d in $DELTAS; do for m in $MUS; do
    echo "python3 bifurcate.py cell egate $d $m --N $N --T $T --sigma $SIG --gate_phase"
    echo "python3 bifurcate.py cell cos   $d $m --N $N --T $T --sigma $SIG --gate_phase"
  done; done
  # diff reference: delta=0.1 row (plain coupling => fusion at low delta/mu)
  for m in $MUS; do
    echo "python3 bifurcate.py cell diff 0.1 $m --N $N --T $T --sigma $SIG --gate_phase"
  done
  # iso reference: one mu per delta (mu irrelevant at mu=0; here mu kept for key, but iso ignores it)
  for d in $DELTAS; do
    echo "python3 bifurcate.py cell iso $d 0.0 --N $N --T $T --sigma $SIG --gate_phase"
  done
}

echo "=== bifurcate sweep: $(emit | wc -l) cells, N=$N T=$T sigma=$SIG ==="
emit | xargs -P 4 -I CMD bash -c CMD
echo "=== BIFURCATE SWEEP DONE ==="
