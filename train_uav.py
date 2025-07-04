#/home/fsc-jupiter/source/Mark/AER1810/on-policy/train_uav.py
import argparse
import os
import yaml
import torch
from pathlib import Path

# on-policy imports
from typing import Union
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.uav_defense.uav_defense_env import UAVDefenseEnv
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPOConfig, RMAPPOLearner
from onpolicy.runner.shared.uav_runner import UAVRunner

# ----------------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parent
CFG_DIR = ROOT / "cfgs"


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a Python dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_fn(env_cfg: dict, seed: int):
    """Return a zero-arg initializer for a UAVDefenseEnv."""
    def _init():
        kwargs = env_cfg.copy()
        kwargs.pop("n_rollout_threads", None)
        kwargs.pop("use_eval", None)
        kwargs.pop("n_eval_rollout_threads", None)
        kwargs["seed"] = seed
        env = UAVDefenseEnv(**kwargs)
        return env
    return _init


def make_vec_env(env_cfg: dict) -> Union[SubprocVecEnv, DummyVecEnv]:
    """
    Instantiate either a DummyVecEnv (for 1 worker) or
    SubprocVecEnv (for >1) given the n_rollout_threads in env_cfg.
    """
    cfg = env_cfg.copy()
    n_rollout = cfg.pop("n_rollout_threads")
    base_seed = cfg.get("seed", 0)
    fns = [make_env_fn(cfg, base_seed + i * 1000) for i in range(n_rollout)]
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
    num_agents = env_cfg["n_def"]  # must match your YAML key in env_defense.yaml

    # ------------------------------------------------------------------------
    # 4) Build R-MAPPO learner
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo_cfg = R_MAPPOConfig(
        **algo_cfg_dict,
        obs_space=obs_space,
        act_space=act_space,
        num_agents=num_agents,
        device=device,
        env_name="UAVDefense",
    )
    learner = RMAPPOLearner(algo_cfg)

    # ------------------------------------------------------------------------
    # 5) Construct runner and kickoff training
    runner = UAVRunner({
        "envs": envs,
        "eval_envs": eval_envs,
        "all_args": algo_cfg,
        "num_agents": num_agents,
        "device": device,
        "env_name": "UAVDefense",
        "run_dir": ROOT,
    })
    runner.run()

    # ------------------------------------------------------------------------
    # 6) Save final checkpoint
    ckpt_dir = ROOT / "checkpoints" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    learner.save(ckpt_dir / "final.pt")
    print(f"[train_uav] 训练结束，模型已保存至 {ckpt_dir}")


if __name__ == "__main__":
    main()
