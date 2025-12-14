#play_train.py
from stable_baselines3 import PPO
from sumo_env import SumoIntersectionEnv

REWARD_MODE = "hybrid"
MODEL_PATH = f"ppo_ppo_sumo_hybrid_prob_turnlanes" # Change to play new models
# reminder to change naming scheme

env = SumoIntersectionEnv(
    sumo_cfg="simple_intersection/probabilistic.sumocfg",
    use_gui=True,
    delta_time=3,
    gui_delay=0.1,
    max_steps=500,
    reward_mode=REWARD_MODE,
)

model = PPO.load(MODEL_PATH)

obs, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(int(action))

env.close()
