# SQ1 SUMO scenario (placeholder)


## What you must add (not included)
- `assets/sumo_networks/sq1/sq1.net.xml` : the SUMO network for SQ1 (user-provided).
- Replace `sq1.rou.xml` with actual OD demand streams (A/B/C/D) matching your paper setup.

## How this repo uses it
The wrapper `src/gamarl/envs/sumo_env.py` expects a `.sumocfg` path from config:
- `env.sumo_cfg_path: assets/scenarios/sq1/sq1.sumocfg`

Then it will start SUMO (via TraCI) and build observations `y=(F,A,m)`.

#
