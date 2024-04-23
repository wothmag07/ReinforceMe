import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class QLearning_Lambda:
    def __init__(self, grid_world, discount_factor=0.95, timesteps=100, episodes=100, evaluation_results=False, lambd=0.9):
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.timesteps = timesteps
        self.episodes = episodes
        self.evaluation_results = evaluation_results
        self.lambd = lambd
        self.learning_rate = None
        self.epsilon_start = None
        self.epsilon_min = None
        self.epsilon_decay = None

    def choose_action(self, state, q_table, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.grid_world.num_actions)
        else:
            return np.argmax(q_table[state])

    def run_episode(self, q_table, epsilon):
        state = 0  # Starting state
        total_reward = 0

        #Initialise eligibility traces
        eligibility_trace = np.zeros_like(q_table)

        for _ in range(self.timesteps):
            action = self.choose_action(state, q_table, epsilon)
            valid_next_states = [next_state for next_state, prob in enumerate(self.grid_world.transition_probs[state, action]) if prob > 0]
            probabilities = [prob for prob in self.grid_world.transition_probs[state, action] if prob > 0]

            next_state = np.random.choice(valid_next_states, p=probabilities)

            reward = self.grid_world.rewards[next_state]

            # Update eligibility trace
            eligibility_trace *= self.discount_factor * self.lambd
            eligibility_trace[state, action] += 1

            td_error = reward + self.discount_factor * np.max(q_table[next_state]) - q_table[state, action]

            # Update Q-values
            q_table[state, action] += self.learning_rate * td_error * eligibility_trace[state, action]

            total_reward += reward

            if next_state == self.grid_world.num_states - 1:  # Reached the goal state
                break

            state = next_state

        return total_reward

    def run(self, learning_rate=0.15, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        all_rewards_per_trial = []  # To store rewards for each trial

        q_table = np.zeros((self.grid_world.num_states, self.grid_world.num_actions))

        for _ in range(100):  # Run 100 trials
            rewards_per_episode = []  # To store rewards for each episode in a trial

            epsilon = self.epsilon_start
            state = 0  # Resetting the state for each trial

            for episode in range(self.episodes):
                state = 0
                total_rewards = self.run_episode(q_table, epsilon)
                rewards_per_episode.append(total_rewards)

                if epsilon > self.epsilon_min:
                    epsilon *= self.epsilon_decay
                    epsilon = max(epsilon, self.epsilon_min)

            all_rewards_per_trial.append(rewards_per_episode)

        all_rewards_per_trial = np.array(all_rewards_per_trial)

        print("Shape of all_rewards_per_trial:", all_rewards_per_trial.shape)


        # Compute mean rewards and standard deviations across trials for each episode
        mean_rewards_across_trials = np.mean(all_rewards_per_trial, axis=1)
        std_rewards_across_trials = np.std(all_rewards_per_trial, axis=1)

        # print(mean_rewards_across_trials.shape)
        # print(std_rewards_across_trials.shape)

        # Plot mean rewards with error bars

        # plt.errorbar(episodes, mean_rewards_across_trials, yerr=std_rewards_across_trials, label='Q-Learning(λ)', fmt='-o', color='r', ecolor='k', capsize=3)
        # plt.xlabel('Episodes')
        # plt.ylabel('Mean Rewards')
        # plt.title('Mean Rewards Over Episodes with Error Bars')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        # # average_rewards = np.mean(self.total_rewards, axis=0)
        # # std_rewards = np.std(self.total_rewards, axis=0)
        episodes = np.arange(1, self.episodes + 1)

        
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=np.max(std_rewards_across_trials))
        colors = cmap(norm(std_rewards_across_trials))
        
        plt.plot(episodes, mean_rewards_across_trials, color='blue', label='Average Reward')
        plt.fill_between(episodes, mean_rewards_across_trials - std_rewards_across_trials, mean_rewards_across_trials + std_rewards_across_trials, color=colors)
        
        # Add color bar legend
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Standard Deviation')

        plt.title('Average Rewards per Episode with Standard Deviation - QLearning_lambda')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.show()

        return mean_rewards_across_trials


grid_world = GridWorld(16, 4)
grid_world.compute_transition_prob()
qlearningAgent = QLearning_Lambda(grid_world)
mean_sarsa_rewards = qlearningAgent.run(learning_rate=0.15, epsilon_start=0.1)
