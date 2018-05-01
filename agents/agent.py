from agents.actor import Actor
from agents.critic import Critic
from agents.ou_noise import OUNoise
from agents.replay_buffer import ReplayBuffer
import numpy as np


class DDPG(object):
    """
    Reinforcement Learning agent that learns using DDPG.深度确定性策略梯度 (DDPG)
    """

    def __init__(self, task, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                 learning_rate=1e-3)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
                                  learning_rate=1e-3)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, learning_rate=1e-4)
        self.critic_target = Critic(self.state_size, self.action_size, learning_rate=1e-4)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(size=self.action_size)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size  # 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # 0.99 discount factor
        self.tau = tau  # 0.01 for soft update of target parameters

        # 初始化
        self.last_state = None
        self.total_reward = 0.0

        # Score tracker and learning parameters
        self.score = 0
        self.best_score = -np.inf
        self.count = 0

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.total_reward = 0.0
        self.count = 0
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        self.total_reward += reward
        self.count += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(states, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,
                                                                                                        self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                      (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        # track best score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
