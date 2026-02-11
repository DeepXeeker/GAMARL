from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig
import torch

from gamarl.envs.synthetic_env import SyntheticTINEnv, SyntheticParams
from gamarl.envs.sumo_env import SumoTINEnv, SumoScenario
from gamarl.models.gamarl_model import GAMARLModel
from gamarl.utils.checkpointing import load_checkpoint
from gamarl.evaluation.evaluator import Evaluator
from gamarl.evaluation.report import save_report


def _make_env(cfg: DictConfig, action_dim: int):
    df = int(cfg.model.df)
    if cfg.env.name == "synthetic":
        params = SyntheticParams(
            episode_horizon=int(cfg.env.episode_horizon),
            decision_interval_s=int(cfg.env.decision_interval_s),
            traffic_load_range=tuple(cfg.env.traffic_load_range),
            adjacency=str(cfg.env.adjacency),
        )
        return SyntheticTINEnv(
            n_intersections=int(cfg.env.n_intersections),
            df=df,
            action_dim=action_dim,
            params=params,
        )

    if cfg.env.name == "sumo":
        scen = SumoScenario(
            sumo_binary=str(cfg.env.sumo_binary),
            sumo_config=str(cfg.env.sumo_config),
            network_file=getattr(cfg.env, "network_file", None),
            route_file=getattr(cfg.env, "route_file", None),
        )
        return SumoTINEnv(
            scenario=scen,
            n_intersections=int(cfg.env.get("n_intersections", 10)),
            df=df,
            action_dim=action_dim,
            episode_horizon=int(cfg.env.episode_horizon),
        )

    raise ValueError(f"Unknown env: {cfg.env.name}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    action_dim = len(cfg.action.phases) * len(cfg.action.durations_s)

    env = _make_env(cfg, action_dim)

    model = GAMARLModel(
        df=int(cfg.model.df),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_transformer_layers=int(cfg.model.n_transformer_layers),
        history_len=int(cfg.model.history_len),
        gate_hidden=int(cfg.model.gate_hidden),
        attn_hidden=int(cfg.model.attn_hidden),
        gumbel_tau=float(cfg.model.gumbel_tau),
        hard_gates=bool(cfg.model.hard_gates),
        gcn_hidden=int(cfg.model.gcn_hidden),
        gcn_activation=str(cfg.model.gcn_activation),
        q_hidden=int(cfg.model.q_hidden),
        action_dim=action_dim,
    ).to(device)

    ckpt_path = os.path.join(os.getcwd(), cfg.log_dir, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, map_location=str(device))
        model.load_state_dict(ckpt["model"])

    model.eval()

    evaluator = Evaluator(env, model, device=device, history_len=int(cfg.model.history_len))
    res = evaluator.run(episodes=5)

    out_path = os.path.join(os.getcwd(), cfg.log_dir, "eval_report.json")
    save_report(out_path, res.metrics)
    print(res.metrics)


if __name__ == "__main__":
    main()
