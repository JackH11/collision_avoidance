from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from gym_env import MovingAvoidanceEnv
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.utils import get_schedule_fn


class ActionLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, action_space_size=None):
        super(ActionLoggingCallback, self).__init__(verbose)
        self.action_counts = {}  # Track action frequencies per episode
        self.episode_actions = []  # Store actions for current episode
        self.episode_count = 0
        self.action_space_size = action_space_size  # Optional: Size of action space for histograms

    def _on_step(self) -> bool:
        # Get action from locals (for vectorized envs, actions is a list)
        action = self.locals.get("action") or self.locals.get("actions")[0]
        step = self.num_timesteps
        self.episode_actions.append(action)

        # Log individual action as a scalar
        self.logger.record("action/action_taken", action)

        # Check if episode ended
        info = self.locals.get("infos")[0]
        if "episode" in info:
            self.episode_count += 1
            # Calculate action frequencies for the episode
            unique, counts = np.unique(self.episode_actions, return_counts=True)
            action_freq = dict(zip(unique, counts))

            # Log action frequencies as scalars
            for action in range(self.action_space_size or max(unique) + 1):
                count = action_freq.get(action, 0)
                self.logger.record(f"action/freq_action_{action}", count)

            # Log actions as a histogram (for distribution visualization)
            self.logger.record("action/action_distribution", np.array(self.episode_actions))

            # Reset episode actions
            self.episode_actions = []

        return True



# Wrap the custom env with make_vec_env
env = make_vec_env(lambda: MovingAvoidanceEnv(), n_envs=1)

#model = DQN.load("dqn_avoidance_agent", env=env)
# Create DQN model

model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=100,
    verbose=1,
    tensorboard_log="./dqn_tensorboard/",
    exploration_fraction=0.9,
    exploration_initial_eps=0.5,
    exploration_final_eps=0.1,
)

#model.exploration_rate = 0.3
#model.exploration_schedule = get_schedule_fn(0.3)  # constant schedule # set initial value

action_callback = ActionLoggingCallback(action_space_size=env.action_space.n)

# Train for 100,000 timesteps
model.learn(total_timesteps=50_000,tb_log_name="dqn",callback=action_callback)

# Save model
model.save("dqn_avoidance_agent")
