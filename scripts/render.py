#/home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/scripts/render.py
import torch
from pathlib import Path
from onpolicy.envs.uav_defense.uav_defense_env import UAVDefenseEnv
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPOConfig, RMAPPOLearner
import yaml

def main():
    # 1) 加载配置
    root = Path(__file__).parent.parent.resolve()
    env_cfg = yaml.safe_load(open(root/"cfgs/env_defense.yaml"))
    algo_cfg = yaml.safe_load(open(root/"cfgs/mappo_defense.yaml"))
    # 2) 创建环境
    env = UAVDefenseEnv(**env_cfg)
    # 3) 加载模型
    cfg = R_MAPPOConfig(**algo_cfg,
                       obs_space=env.observation_space,
                       act_space=env.action_space,
                       num_agents=env.n_def,
                       device=torch.device("cpu"))
    learner = RMAPPOLearner(cfg)
    learner.restore(root/"checkpoints"/"uav_defense"/"final.pt")
    # 4) 渲染若干回合
    for ep in range(5):
        obs, _ = env.reset()
        done = [False]*env.n_def
        while not all(done):
            # 多智能体：依次决策
            acts = learner.policy.act(obs, deterministic=True)
            obs, _, done, _ = env.step(acts)
            env.render()
    env.close()

if __name__=="__main__":
    main()
