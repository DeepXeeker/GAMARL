# GAMARL - Gated-Attention Graph MARL for Corridor Traffic Signal Control

**Selective Inter-Intersection Communication for Partially Observable Corridor Traffic Control via Gated-Attention Graph MARL**  
Hamza Mukhtar

It implements:
- **Two-level Traffic Intersection Network (TIN)** with a local **Distributed Interaction Graph (DIG)** and a corridor-level **Central Cooperation Graph (CCG)**.
- Local **Transformer temporal encoder** + **hard gating + attention-weighted fusion** (Eqs. 6–9 in the paper) for selective inter-intersection communication.
- Corridor-level **degree-normalized masked graph fusion** (Eqs. 10–12) feeding a **value-based Q-network** (Eq. 13), trained with replay + target network (Eq. 14).
- A **progression-oriented reward** that penalizes downstream waiting after corridor entry (Eqs. 4–5).
- Curriculum growth for scalability (start with M0=2 and add one intersection every 50 episodes).

---

## Quickstart (Synthetic)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/train.sh configs/env_synthetic.yaml
bash scripts/eval.sh configs/env_synthetic.yaml
```

---

## SUMO Setup

To run SUMO scenarios you need:
1) SUMO installed and `SUMO_HOME` set, or `sumo` available on PATH.
2) Provide network + route assets under:
- `assets/sumo_networks/`
- `assets/scenarios/`

Example:
```bash
export SUMO_HOME=/path/to/sumo
bash scripts/train.sh configs/env_sq1.yaml
```

---

## Repo structure

- `src/gamarl/envs/` — synthetic + SUMO environment wrappers and observation builder
- `src/gamarl/models/` — Transformer encoder, gated-attention communication, GCN fusion, Q heads
- `src/gamarl/algo/` — DQN training loop, replay buffer, curriculum
- `src/train.py` / `src/eval.py` — runnable entrypoints
- `configs/` — default + env configs
- `tests/` — shape + gating + GCN normalization tests

---

## Citation

See `CITATION.cff`.

---

## License

MIT (see `LICENSE`).
