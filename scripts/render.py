import argparse
import os
from pathlib import Path
import torch
import yaml
from onpolicy.envs.uav_defense.uav_defense_env import UAVDefenseEnv
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPOConfig, RMAPPOLearner

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
    env_cfg = yaml.safe_load(open(root / "cfgs" / "env_defense.yaml"))
    algo_cfg = yaml.safe_load(open(root / "cfgs" / "mappo_defense.yaml"))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path) if args.model_path else \
        root / "checkpoints" / args.exp_name / "final.pt"

    env = UAVDefenseEnv(**env_cfg)
    cfg = R_MAPPOConfig(
        **algo_cfg,
        obs_space=env.observation_space,
        act_space=env.action_space,
        num_agents=env.n_def,
        device=device,
    )
    learner = RMAPPOLearner(cfg)
    learner.restore(root/"checkpoints"/"uav_defense"/"final.pt")
    learner.restore(model_path)

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = [False] * env.n_def
        while not all(done):
            # 多智能体：依次决策
            acts = learner.policy.act(obs, deterministic=True)
            obs, _, done, _ = env.step(acts)
            env.render()
    env.close()


if __name__ == "__main__":
    main()
