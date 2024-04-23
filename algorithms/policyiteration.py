import numpy as np
from gridworld import GridWorld
import matplotlib.pyplot as plt

class PolicyIteration(GridWorld):
    def __init__(self, num_states, num_actions, gamma):
        super().__init__(num_states, num_actions)
        self.gamma = gamma

    def policy_iteration(self, epi=0.7):
        self.compute_transition_prob()

        policy = np.random.randint(0, self.num_actions, self.num_states)
        policy_stable = False

        while not policy_stable:
            # Policy evaluation
            v = np.zeros(self.num_states)
            while True:
                delta = 0
                for s in range(self.num_states):
                    v_old = v[s]
                    a = policy[s]
                    temp_sum = sum(self.transition_probs[s, a, s1] * (self.rewards.get(s, 0) + self.gamma * v[s1]) for s1 in self.next_states[s][a])
                    v[s] = temp_sum
                    delta = max(delta, abs(v_old - v[s]))
                if delta < 1e-7:
                    break

            # Policy improvement
            policy_stable = True
            for s in range(self.num_states):
                old_action = policy[s]
                q = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    temp_sum = sum(self.transition_probs[s, a, s1] * (self.rewards.get(s, 0) + self.gamma * v[s1]) for s1 in self.next_states[s][a])
                    q[a] = temp_sum
                policy[s] = np.argmax(q)
                if old_action != policy[s]:
                    policy_stable = False

        print("Convergence reached.")
        # Plot
        plt.plot(v)
        plt.title('Optimal Value Function')
        plt.xlabel('States')
        plt.ylabel('Value')
        plt.show()

        return policy, v

if __name__ == "__main__":
    num_states = 16
    actions = 4
    gamma = 0.95

    pi = PolicyIteration(num_states=num_states, num_actions=actions, gamma=gamma)
    policy, v = pi.policy_iteration()

    print("Optimal Policy : {}".format(policy))
    print("Values : {}".format(v))
