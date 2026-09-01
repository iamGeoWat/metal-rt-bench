#!/bin/bash
# Build + run both programs and keep the raw output under results/. Needs the macOS 26 SDK (Xcode 26/27
# or its Command Line Tools) and an Apple-silicon Mac; hardware RT needs an Apple9+ GPU (M3/A17 Pro or later).
set -euo pipefail
cd "$(dirname "$0")"
clang -fobjc-arc -O2 -mmacosx-version-min=26.0 rtprobe.m -framework Foundation -framework Metal -framework MetalFX -o rtprobe
clang -fobjc-arc -O2 -mmacosx-version-min=26.0 rtbench.m -framework Foundation -framework Metal -o rtbench
dev=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2; exit}' | tr ' ' '-')
stamp=$(date +%Y-%m-%d)
out="results/${stamp}-${dev:-unknown}.txt"
{
	echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)  $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))  $(clang --version | head -1)"
	echo "# power: $(pmset -g batt | head -1 | sed 's/^[^'"'"']*//')"
	echo; echo "== rtprobe =="; ./rtprobe
	for seg in 512 1024; do echo; echo "== rtbench $seg =="; ./rtbench "$seg"; done
} | tee "$out"
echo "saved $out"
