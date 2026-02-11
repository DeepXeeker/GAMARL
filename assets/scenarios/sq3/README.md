# SQ3 SUMO scenario (placeholder)

This folder provides *template* SUMO configuration files for **SQ3**.

## What you must add (not included)
- `assets/sumo_networks/sq3/sq3.net.xml` : the SUMO network for SQ3 (user-provided).
- Replace `sq3.rou.xml` with actual OD demand streams (A/B/C/D) matching your paper setup.

## How this repo uses it
The wrapper `src/gamarl/envs/sumo_env.py` expects a `.sumocfg` path from config:
- `env.sumo_cfg_path: assets/scenarios/sq3/sq3.sumocfg`

Then it will start SUMO (via TraCI) and build observations `y=(F,A,m)`.

## Notes
These templates intentionally do not include proprietary networks or datasets.
