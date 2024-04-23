import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class ActorCritic:
    def __init__(self, num_states, num_actions, gamma, alpha_critic, alpha_actor, lambda_val):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.alpha_critic = alpha_critic
        self.alpha_actor = alpha_actor
        self.lambda_val = lambda_val

        # Initialize values and policies for all states-action pairs to zero
        self.value_function = np.zeros(num_states)
        self.policy = np.ones((num_states, num_actions)) / num_actions

        #Initialise eligibility traces for both critic and actor
        self.critic_eligibility = np.zeros(num_states)
        self.actor_eligibility = np.zeros((num_states, num_actions))
        
        self.total_rewards = None

    def select_action(self, state):
        return np.argmax(self.policy[state])

    def update(self, state, action, reward, next_state, done):
        td_error = reward + self.gamma * self.value_function[next_state] - self.value_function[state]

        # Update eligibility traces
        self.critic_eligibility *= self.lambda_val * self.gamma
        self.critic_eligibility[state] += 1
        self.actor_eligibility *= self.lambda_val * self.gamma
        self.actor_eligibility[state, action] += 1

        # Update value function
        self.value_function += self.alpha_critic * td_error * self.critic_eligibility

        # Update policy
        advantages = np.zeros(self.num_actions)
        advantages[action] = td_error
        self.policy[state] += self.alpha_actor * advantages * self.actor_eligibility[state]

        if done:
            self.critic_eligibility.fill(0)
            self.actor_eligibility.fill(0)

        return td_error

    def run_trials(self, env, num_trials, num_episodes):
        self.total_rewards = np.zeros((num_trials, num_episodes))
        for trial in range(num_trials):
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done = env.step(state, env.actions[action])
                    td_error = self.update(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                self.total_rewards[trial, episode] = episode_reward
        return self.total_rewards
    
    def plot_rewards(self):
        average_rewards = np.mean(self.total_rewards, axis=0)
        std_rewards = np.std(self.total_rewards, axis=0)
        episodes = np.arange(1, len(average_rewards) + 1)
        
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=np.max(std_rewards))
        colors = cmap(norm(std_rewards))
        
        plt.plot(episodes, average_rewards, color='blue', label='Average Reward')
        plt.fill_between(episodes, average_rewards - std_rewards, average_rewards + std_rewards, color=colors)
        
        # Add color bar legend
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Standard Deviation')

        plt.title('Average Rewards per Episode with Standard Deviation')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.show()


grid_world = GridWorld(16, 4)
grid_world.compute_transition_prob()
ac_agent = ActorCritic(num_states=16, num_actions=4, gamma=0.95, alpha_critic=0.001, alpha_actor=0.001, lambda_val=0.9)
rewards = ac_agent.run_trials(grid_world, num_trials=10, num_episodes=100)
ac_agent.plot_rewards()
