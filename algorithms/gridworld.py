import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class GridWorld:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.actions = ['left', 'up', 'right', 'down']
        self.transition_probs = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.rewards = self.initialize_rewards()
        self.next_states = collections.defaultdict(dict)
        self.compute_transition_prob()

    def get_next_state(self, state, action):
        num_rows, num_columns = 4, 4

        row, col = divmod(state, num_columns)

        if action == 'left' and col > 0:  # Left
            return state - 1
        elif action == 'up' and row > 0:  # Up
            return state - num_columns
        elif action == 'right' and col < num_columns - 1:  # Right
            return state + 1
        elif action == 'down' and row < num_rows - 1:  # Down
            return state + num_columns
        else:
            return state

    def reset(self):      
      # Return the initial state
      return 0

    def step(self, state, action):
        next_state = self.get_next_state(state, action)
        reward = self.rewards[next_state]
        done = (next_state == self.num_states - 1)  # Check if goal state reached
        return next_state, reward, done

    def compute_transition_prob(self):
        for state in range(self.num_states):
            for action_label in self.actions:
                action = self.actions.index(action_label)
                if action_label == 'left' or action_label == 'right':  # Left or Right
                    fail_states = (1, 3)
                elif action_label == 'up' or action_label == 'down':  # Up or Down
                    fail_states = (0, 2)

                success_next_s = self.get_next_state(state, action_label)
                fail_next_s1 = self.get_next_state(state, self.actions[fail_states[0]])
                fail_next_s2 = self.get_next_state(state, self.actions[fail_states[1]])

                states = [success_next_s, fail_next_s1, fail_next_s2]
                states = list(dict.fromkeys(states))

                if len(states) == 3:
                    # Transition probabilities
                    self.transition_probs[state, action, states[0]] = 0.8
                    self.transition_probs[state, action, states[1]] = 0.1
                    self.transition_probs[state, action, states[2]] = 0.1
                elif len(states) == 2:
                    self.transition_probs[state, action, states[0]] = 0.9
                    self.transition_probs[state, action, states[1]] = 0.1

                self.next_states[state][action] = states

    def initialize_rewards(self):
        rewards = {}
        for state in range(self.num_states):
            if state == 9 or state == 10:  # water states
                rewards[state] = -5
            elif state == 1 or state == 2:  # wildfire states
                rewards[state] = -10
            else:
                rewards[state] = -1
        rewards[self.num_states - 1] = 100  # goal state
        return rewards
