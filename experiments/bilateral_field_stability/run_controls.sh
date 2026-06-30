#!/usr/bin/env bash
# Orchestrate all control arms with 4-way CPU parallelism.
# Wave 1: all base arms (incl. bilatL, which measures the match amplitude).
# Wave 2: matched-noise arms (white/colored) -- depend on bilatL's measured amp.
set -u
cd "$(dirname "$0")"
N=${N:-60}; T=${T:-800}
export N T

run() { echo "python3 controls.py $1 --N $N --T $T"; }

echo "=== WAVE 1 (base arms, 4-way parallel) ==="
{
  run "single 0.006"
  run "single 0.007"
  run "single 0.008"
  run "single 0.0075"
  run "single 0.0085"
  run "single 0.009"
  run "bilatL 0.1 0.007"
  run "sham_frozen 0.1 0.007"
  run "sham_indep 0.1 0.007"
  run "sham_mutual 0.1 0.007"
  run "kick_lock 0.007"
  run "kick_alive 0.007"
} | xargs -P 4 -I CMD bash -c CMD

echo "=== WAVE 2 (matched noise, needs bilatL amp) ==="
{
  run "white 0.1 0.007"
  run "colored 0.1 0.007"
} | xargs -P 4 -I CMD bash -c CMD

echo "=== ALL CONTROL ARMS DONE ==="
