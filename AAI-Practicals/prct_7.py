import numpy as np

states = 5
actions = 2
Q = np.zeros((states, actions))

episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.1

for _ in range(episodes):
    state = 0
    while state < states - 1:
        if np.random.rand() < epsilon:
            action = np.random.randint(actions)
        else:
            action = np.argmax(Q[state])

        next_state = state + 1 if action == 1 else state
        reward = 1 if next_state == states - 1 else 0

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Learned Q-table:")
print(Q)
