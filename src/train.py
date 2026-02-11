from __future__ import annotations

import os
from dataclasses import asdict

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

from gamarl.utils.seed import set_seed
from gamarl.utils.logger import make_logger
from gamarl.utils.schedulers import LinearSchedule
from gamarl.utils.checkpointing import make_ckpt_paths, save_checkpoint

from gamarl.envs.synthetic_env import SyntheticTINEnv, SyntheticParams
from gamarl.envs.sumo_env import SumoTINEnv, SumoScenario

from gamarl.models.gamarl_model import GAMARLModel
from gamarl.algo.dqn_agent import DQNAgent, DQNConfig
from gamarl.algo.curriculum import Curriculum
from gamarl.data.replay_buffer import ReplayBuffer
from gamarl.data.batch import Transition


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
        # n_intersections is scenario-dependent; user must set it in env config or code.
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
    set_seed(int(cfg.seed))

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    action_dim = len(cfg.action.phases) * len(cfg.action.durations_s)

    run_dir = os.getcwd()  # hydra changes cwd
    log = make_logger(os.path.join(run_dir, cfg.log_dir), name="train")
    log.logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

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
    )

    target = GAMARLModel(
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
    )
    target.load_state_dict(model.state_dict())

    eps = LinearSchedule(
        start=float(cfg.training.epsilon.start),
        end=float(cfg.training.epsilon.end),
        duration=int(cfg.training.epsilon.anneal_steps),
    )

    agent = DQNAgent(
        model=model,
        target=target,
        action_dim=action_dim,
        device=device,
        cfg=DQNConfig(
            gamma=float(cfg.training.gamma),
            lr=float(cfg.training.lr),
            target_update_steps=int(cfg.training.target_update_steps),
            warmup_steps=int(cfg.training.warmup_steps),
            batch_size=int(cfg.training.batch_size),
        ),
        eps=eps,
    )

    buffer = ReplayBuffer(int(cfg.training.buffer_size))
    curriculum = Curriculum(
        enabled=bool(cfg.curriculum.enabled),
        m0=int(cfg.curriculum.m0),
        add_every_episodes=int(cfg.curriculum.add_every_episodes),
    )

    ckpts = make_ckpt_paths(os.path.join(run_dir, cfg.log_dir))

    total_steps = int(cfg.training.total_steps)
    episode = 0
    best_reward = -1e18

    while agent.steps < total_steps:
        F, A, m, info = env.reset(seed=episode)
        done = False
        ep_reward = 0.0

        # curriculum: restrict active intersections by masking
        active = curriculum.active_count(episode, env.n_intersections)
        m = m.copy()
        m[active:] = 0.0

        while not done and agent.steps < total_steps:
            Ft = torch.tensor(F, dtype=torch.float32, device=device).unsqueeze(0)
            At = torch.tensor(A, dtype=torch.float32, device=device).unsqueeze(0)
            mt = torch.tensor(m, dtype=torch.float32, device=device).unsqueeze(0)

            F_hist = Ft.unsqueeze(2).repeat(1, 1, int(cfg.model.history_len), 1)
            Q, _ = agent.model(F_hist, Ft, At, mt)

            act = agent.act(Q)
            # apply curriculum mask: inactive nodes keep action 0
            act = act.copy()
            act[active:] = 0

            out = env.step(act)

            # store transition
            t = Transition(
                F=torch.tensor(F, dtype=torch.float32),
                A=torch.tensor(A, dtype=torch.float32),
                m=torch.tensor(m, dtype=torch.float32),
                action=torch.tensor(act, dtype=torch.long),
                reward=torch.tensor([out.reward], dtype=torch.float32),
                F_next=torch.tensor(out.F, dtype=torch.float32),
                A_next=torch.tensor(out.A, dtype=torch.float32),
                m_next=torch.tensor(out.m, dtype=torch.float32),
                done=torch.tensor([1.0 if out.done else 0.0], dtype=torch.float32),
            )
            buffer.add(t)

            upd = agent.update(buffer)
            if upd:
                log.tb.add_scalar("train/loss", upd["loss"], agent.steps)
                log.tb.add_scalar("train/epsilon", upd["epsilon"], agent.steps)

            F = out.F
            done = out.done
            ep_reward += out.reward

        log.logger.info("Episode %d reward=%.4f active=%d steps=%d", episode, ep_reward, active, agent.steps)
        log.tb.add_scalar("train/episode_reward", ep_reward, episode)

        # checkpoints
        save_checkpoint(ckpts.latest, {"model": agent.model.state_dict(), "steps": agent.steps, "cfg": OmegaConf.to_container(cfg)})
        if ep_reward > best_reward:
            best_reward = ep_reward
            save_checkpoint(ckpts.best, {"model": agent.model.state_dict(), "steps": agent.steps, "cfg": OmegaConf.to_container(cfg)})

        episode += 1


if __name__ == "__main__":
    main()
