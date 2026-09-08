#!/usr/bin/env bash
# Pre-flight for any timing run. An orphaned process holding a NeuronCore fakes a
# mid-run throughput collapse, and a bimodal step-time trace is a contention
# signature until proven otherwise. Run this BEFORE you believe a number.
set -u

echo "== instance =="
neuron-ls 2>/dev/null | head -5 || echo "neuron-ls not on PATH"

echo
echo "== device holders =="
holders=0
for d in $(seq 0 15); do
    if [ -e "/dev/neuron$d" ]; then
        n=$(lsof "/dev/neuron$d" 2>/dev/null | tail -n +2 | wc -l)
        holders=$((holders + n))
        [ "$n" -gt 0 ] && lsof "/dev/neuron$d" 2>/dev/null | tail -n +2 | awk -v d="$d" '{print "  neuron"d": "$1" pid "$2" ("$3")"}'
    fi
done

if [ "$holders" -eq 0 ]; then
    echo "  none — clear to run"
else
    echo
    echo "  $holders open handle(s). If none of these is your job, kill them before timing."
fi

echo
echo "== compiler cache =="
for v in TORCH_NEURONX_HLO_CACHE_DIR TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR \
         TORCH_NEURONX_NEFF_CACHE_DIR TORCH_NEURONX_DEBUG_DIR; do
    printf '  %-38s %s\n' "$v" "${!v:-<unset, will default under /tmp>}"
done
printf '  %-38s %s\n' "NEURON_RT_VISIBLE_CORES" "${NEURON_RT_VISIBLE_CORES:-<unset>}"
printf '  %-38s %s\n' "NEURON_CC_FLAGS" "${NEURON_CC_FLAGS:-<unset>}"
[ -n "${NEURON_CC_FLAGS:-}" ] && echo "    ^ a stale value is served from the NEFF cache; clear the cache before A/B-ing flags"

exit $([ "$holders" -eq 0 ] && echo 0 || echo 1)
