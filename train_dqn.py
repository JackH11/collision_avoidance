from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from moving_env import MovingAvoidanceEnv

# Wrap the custom env with make_vec_env
env = make_vec_env(MovingAvoidanceEnv, n_envs=1)

# Create DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=100,
    verbose=1
)

# Train for 100,000 timesteps
model.learn(total_timesteps=100_000)

# Save model
model.save("dqn_avoidance_agent")
