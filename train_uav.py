#/home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/setup.py
"""
train_uav.py
============

● 读取 cfgs/env_defense.yaml & cfgs/mappo_defense.yaml
● 使用 SubprocVecEnv / DummyVecEnv 并行化环境
● 选择 r-MAPPO (recurrent MAPPO) 算法
● 支持 --exp-name, --gpu, --stop-timesteps CLI 参数

Usage:
    bash scripts/train.sh
    # 或者手动:
    python train_uav.py --exp-name myrun --gpu 0 --stop-timesteps 2000000
"""
import argparse
import os
import yaml
import torch
from pathlib import Path

# on-policy imports
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.uav_defense.uav_defense_env import UAVDefenseEnv
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPOConfig, RMAPPOLearner
from onpolicy.runner.shared.base_runner import Runner

# ----------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
CFG_DIR = ROOT / "cfgs"


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_fn(env_cfg: dict, seed: int):
    """
    Return a zero-arg initializer for a UAVDefenseEnv,
    capturing the env_cfg and a unique seed.
    """
    def _init():
        env = UAVDefenseEnv(**env_cfg)
        # PettingZoo-style envs often need a .seed() call
        env.seed(seed)
        return env
    return _init


def make_vec_env(env_cfg: dict) -> SubprocVecEnv | DummyVecEnv:
    """
    Instantiate either a DummyVecEnv (for 1 worker) or
    SubprocVecEnv (for >1) given the n_rollout_threads in env_cfg.
    """
    n_rollout = env_cfg["n_rollout_threads"]
    base_seed = env_cfg["seed"]
    fns = [make_env_fn(env_cfg, base_seed + i * 1000) for i in range(n_rollout)]
    if n_rollout == 1:
        return DummyVecEnv(fns)
    else:
        return SubprocVecEnv(fns)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name",       default="uav_defense",   help="log dir name")
    p.add_argument("--gpu",            default="0",             help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--stop-timesteps", type=int, default=2_000_000, help="训练总步数")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ------------------------------------------------------------------------
    # 1) Load both env & algo configs from YAML
    env_cfg       = load_yaml(CFG_DIR / "env_defense.yaml")
    algo_cfg_dict = load_yaml(CFG_DIR / "mappo_defense.yaml")

    # ------------------------------------------------------------------------
    # 2) Build vectorized train & (optionally) eval envs
    envs      = make_vec_env(env_cfg)
    eval_envs = None
    if env_cfg.get("use_eval", False):
        eval_cfg = env_cfg.copy()
        eval_cfg["n_rollout_threads"] = env_cfg.get("n_eval_rollout_threads", 1)
        eval_envs = make_vec_env(eval_cfg)

    # ------------------------------------------------------------------------
    # 3) Extract obs/action spaces & agent count
    obs_space  = envs.observation_space
    act_space  = envs.action_space
    num_agents = env_cfg["n_defenders"]  # must match your YAML key

    # ------------------------------------------------------------------------
    # 4) Build R-MAPPO learner
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo_cfg = R_MAPPOConfig(
        **algo_cfg_dict,
        obs_space=obs_space,
        act_space=act_space,
        num_agents=num_agents,
        device=device,
    )
    learner = RMAPPOLearner(algo_cfg)

    # ------------------------------------------------------------------------
    # 5) Construct runner and kickoff training
    runner = Runner(envs, learner, algo_cfg, eval_envs=eval_envs)
    runner.run(
        total_timesteps=args.stop_timesteps,
        log_dir=ROOT / "runs" / args.exp_name,
    )

    # ------------------------------------------------------------------------
    # 6) Save final checkpoint
    ckpt_dir = ROOT / "checkpoints" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    learner.save(ckpt_dir / "final.pt")
    print(f"[train_uav] 训练结束，模型已保存至 {ckpt_dir}")


if __name__ == "__main__":
    main()
