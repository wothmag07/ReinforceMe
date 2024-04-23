from gridworld import GridWorld


def generate_episodes(gridworld, policy, num_episodes=30):
    episodes = []
    for _ in range(num_episodes):
        episode = []
        state = 0  # Start state
        while state != gridworld.num_states - 1:  # Goal state
            action = policy[state]
            next_state = gridworld.get_next_state(state, action)
            reward = gridworld.rewards[next_state]
            episode.append((state, action, reward))
            state = next_state
        episode.append((state, None, gridworld.rewards[state]))  # Append the goal state
        episodes.append(episode)
    return episodes

# fixed_policy = {
#     0: 'down', 1: 'right', 2: 'right', 3: 'down',
#     4: 'right', 5: 'left', 6: 'down', 7: 'down',
#     8: 'right', 9: 'down', 10: 'down', 11: 'down',
#     12: 'right', 13: 'right', 14: 'down', 15: None  # None for the goal state
# }


fixed_policy = {
    0: 3, 1: 2, 2: 2, 3: 3,
    4: 2, 5: 0, 6: 3, 7: 3,
    8: 2, 9: 3, 10: 3, 11: 3,
    12: 2, 13: 2, 14: 3, 15: None  # None for the goal state
}

gridworld = GridWorld(num_states=16,  num_actions=4)
# Generate and print all possible episodes with the fixed policy
episodes = generate_episodes(gridworld, fixed_policy)
for i, episode in enumerate(episodes):
    print(f"Episode {i + 1}: {episode}")
