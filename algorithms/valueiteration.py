import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld

class ValueIteration(GridWorld):
    def __init__(self, num_states, num_actions, gamma):
        super().__init__(num_states, num_actions)
        self.gamma = gamma

    def value_iteration(self, epi=0.7):
        self.compute_transition_prob()

        v = np.zeros(self.num_states)
        v_history = [v.copy()]
        delta = 1

        # value iter loop:
        while delta > 1e-7:
            delta = 0
            for s in range(self.num_states):
                v_old = v[s]
                q = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    temp_sum = self.gamma * sum(self.transition_probs[s, a, s1] * v[s1] for s1 in self.next_states[s][a])
                    q[a] = self.rewards.get(s, 0) + temp_sum
                v[s] = np.max(q)
                delta = max(delta, abs(v_old - v[s]))
            v_history.append(v.copy())

        # derive policy
        policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            q = np.zeros(self.num_actions)
            for a in range(self.num_actions):
                temp_sum = self.gamma * sum(self.transition_probs[s, a, s1] * v[s1] for s1 in self.next_states[s][a])
                q[a] = self.rewards.get(s, 0) + temp_sum
            policy[s] = np.argmax(q)

        print("Convergence after {} iterations.".format(len(v_history)))
        # Plot
        plt.plot(v)
        plt.title('Optimal Value Function')
        plt.xlabel('States')
        plt.ylabel('Value')
        plt.show()

        return policy, v



#policy - v*(s)

if __name__ == "__main__":

    num_states = 16
    actions = 4
    gamma = 0.95
    epislon = 1e-6

    vi = ValueIteration(num_states=num_states, num_actions=actions, gamma=gamma)

    policy, v = vi.value_iteration()

    print("Policy : ")
    print(policy)
    print("Values : ")
    print(v)