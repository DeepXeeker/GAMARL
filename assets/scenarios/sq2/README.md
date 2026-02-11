# SQ2 SUMO scenario 


- `assets/sumo_networks/sq2/sq2.net.xml` : the SUMO network for SQ2 (user-provided).
- Replace `sq2.rou.xml` with actual OD demand streams (A/B/C/D) matching your paper setup.

The wrapper `src/gamarl/envs/sumo_env.py` expects a `.sumocfg` path from config:
- `env.sumo_cfg_path: assets/scenarios/sq2/sq2.sumocfg`

Then it will start SUMO (via TraCI) and build observations `y=(F,A,m)`.
