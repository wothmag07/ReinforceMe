import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld

class SARSA:
    def __init__(self, grid_world, discount_factor=0.95, timesteps=100, episodes=100, evaluation_results=False):
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.timesteps = timesteps
        self.episodes = episodes
        self.evaluation_results = evaluation_results

    def choose_action(self, state, q_table, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.grid_world.num_actions)
        else:
            return np.argmax(q_table[state])

    def run_episode(self, q_table, epsilon):
        state = 0  # Starting state
        total_reward = 0

        for _ in range(self.timesteps):
            action = self.choose_action(state, q_table, epsilon)
            valid_next_states = [next_state for next_state, prob in enumerate(self.grid_world.transition_probs[state, action]) if prob > 0]
            probabilities = [prob for prob in self.grid_world.transition_probs[state, action] if prob > 0]

            next_state = np.random.choice(valid_next_states, p=probabilities)

            reward = self.grid_world.rewards[next_state]
            next_action = self.choose_action(next_state, q_table, epsilon)
            q_table[state, action] += self.learning_rate * (reward +
                                                             self.discount_factor * q_table[next_state, next_action] -
                                                             q_table[state, action])
            total_reward += reward

            if next_state == self.grid_world.num_states - 1:  # Reached the goal state
                break

            state = next_state

        return total_reward

    def calculate_cumulative_rewards(self, q_table):
        cumulative_rewards = []
        for episode in range(1, self.episodes + 1):
            total_rewards = self.run_episode(q_table, epsilon=0)
            cumulative_rewards.append(total_rewards)
        return cumulative_rewards

    def run(self, learning_rate=0.15, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        all_rewards_per_episode = []  # To store rewards for each episode in all trials
        sarsa_cumulative_rewards = []  # To store cumulative rewards for SARSA
        all_epsilon_values = []

        q_table = np.zeros((self.grid_world.num_states, self.grid_world.num_actions))

        for _ in range(100):  # Run 100 trials
            rewards_per_episode = []  # To store rewards for each episode in a trial
            cumulative_rewards = []  # To store cumulative rewards for each episode in a trial
            epsilon_values = []

            epsilon = self.epsilon_start
            state = 0  # Resetting the state for each trial

            for episode in range(1, self.episodes + 1):
                total_rewards = self.run_episode(q_table, epsilon)
                rewards_per_episode.append(total_rewards)
                cumulative_rewards.append(np.sum(rewards_per_episode))
                epsilon_values.append(epsilon)

                if epsilon > self.epsilon_min:
                    epsilon *= self.epsilon_decay
                    epsilon = max(epsilon, self.epsilon_min)

            all_rewards_per_episode.append(rewards_per_episode)
            sarsa_cumulative_rewards.append(cumulative_rewards)
            all_epsilon_values.append(epsilon_values)

        # Compute mean and standard deviation across trials for each episode
        mean_rewards = np.mean(all_rewards_per_episode, axis=0)
        std_rewards = np.std(all_rewards_per_episode, axis=0)
        mean_sarsa_cumulative_rewards = np.mean(sarsa_cumulative_rewards, axis=0)
        mean_epsilon = np.mean(all_epsilon_values, axis=0)

        # Plotting
        episodes = np.arange(1, self.episodes + 1)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Rewards', color=color)
        ax1.errorbar(episodes, mean_rewards, yerr=std_rewards, label='SARSA', fmt='-o', color='r', ecolor='k', capsize=3)

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Epsilon', color=color)
        ax2.plot(episodes, mean_epsilon, color=color, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(False)  # Turn off grid for the epsilon plot

        fig.tight_layout()
        plt.title('Mean Rewards and Epsilon Decay (SARSA)')
        plt.show()

        # Return the mean cumulative rewards for further analysis
        return mean_sarsa_cumulative_rewards

if __name__ == "__main__":

    grid_world = GridWorld(16, 4)
    grid_world.compute_transition_prob()
    sarsa = SARSA(grid_world)
    mean_sarsa_cumulative_rewards = sarsa.run(learning_rate=0.1, epsilon_start=0.5)