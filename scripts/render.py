import argparse
import os
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import yaml
from onpolicy.envs.uav_defense.uav_defense_env import UAVDefenseEnv
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPOConfig, RMAPPOLearner

ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="uav_defense", help="experiment name")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--model-path", default=None, help="path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="number of episodes")
    parser.add_argument("--eval-only", action="store_true", help="unused placeholder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = yaml.safe_load(open(ROOT / "cfgs" / "env_defense.yaml"))
    algo_cfg = yaml.safe_load(open(ROOT / "cfgs" / "mappo_defense.yaml"))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path) if args.model_path else \
        ROOT / "checkpoints" / args.exp_name / "final.pt"

    env = UAVDefenseEnv(**env_cfg)
    cfg = R_MAPPOConfig(
        **algo_cfg,
        obs_space=env.observation_space,
        act_space=env.action_space,
        num_agents=env.n_def,
        device=device,
    )
    learner = RMAPPOLearner(cfg)
    learner.restore(model_path)

    for _ in range(args.episodes):
        obs, _ = env.reset()
        rnn_states = np.zeros((1, env.n_def, cfg.recurrent_N, cfg.hidden_size), dtype=np.float32)
        masks = np.ones((1, env.n_def, 1), dtype=np.float32)
        done = {a: False for a in env.agents}

        while not all(done.values()):
            obs_array = np.stack([obs[a] for a in env.agents])

            action, rnn_states[0] = learner.policy.act(obs_array, rnn_states[0], masks[0], deterministic=True)
            action = action.detach().cpu().numpy()
            rnn_states = rnn_states.detach().cpu().numpy()

            action_dict: Dict[str, np.ndarray] = {a: act for a, act in zip(env.agents, action)}

            obs, _, terminations, truncations, _ = env.step(action_dict)

            done = {a: terminations[a] or truncations[a] for a in env.agents}
            for i, a in enumerate(env.agents):
                if done[a]:
                    rnn_states[0, i] = 0
                    masks[0, i] = 0
                else:
                    masks[0, i] = 1
            env.render()
    env.close()


if __name__ == "__main__":
    main()
