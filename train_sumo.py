# train_sumo.py

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoIntersectionEnv
from callbacks import EpisodeMetricsCallback

REWARD_MODE = "hybrid" 
RUN_NAME = f"sumo_hybrid_prob_turnlanes"


def make_env():
    return SumoIntersectionEnv(
        sumo_cfg="simple_intersection/probabilistic.sumocfg",
        use_gui=False,
        delta_time=1,          # 1s per step
        max_steps=800,         # steps per episode cap
        reward_mode=REWARD_MODE,
        min_green=10.0,
        max_green=60.0,
        yellow_min=3.0,
        yellow_max=7.0,
    )


env = make_env()

# PPO config
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=20,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=f"./tb_logs_{RUN_NAME}",
)

episode_cb = EpisodeMetricsCallback()
checkpoint_cb = CheckpointCallback(
    save_freq=50_000 // model.n_steps,  # every ~50k steps of data
    save_path=f"./checkpoints_{RUN_NAME}",
    name_prefix="ppo_sumo",
)

model.learn(
    total_timesteps=300_000,    # total steps
    callback=[episode_cb, checkpoint_cb],
)

model.save(f"ppo_{RUN_NAME}")
env.close()

