#!/usr/bin/env bash
set -euo pipefail

mkdir -p assets/sumo_networks assets/scenarios

cat <<'EOF'
Assets are intentionally not included.

Put your SUMO networks and scenarios under:
  assets/sumo_networks/<SCENARIO>/
  assets/scenarios/<SCENARIO>/

Example expected paths (see configs/env_sq*.yaml):
  assets/scenarios/SQ1/sq1.sumocfg
  assets/sumo_networks/SQ1/net.net.xml
  assets/scenarios/SQ1/routes.rou.xml
EOF
