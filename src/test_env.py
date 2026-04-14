import sys
sys.path.append('.')

from env.drone_env import DroneEnv
import numpy as np

env = DroneEnv()

print("Testing environment with random agent...")
print(f"State size: {env.state_size}")
print(f"Action size: {env.action_size}")

total_rewards = []

for episode in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = np.random.randint(0, env.action_size)
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

    total_rewards.append(total_reward)
    print(f"Episode {episode+1:2d} | Steps: {steps:4d} | Reward: {total_reward:8.1f} | Final state: x={state[0]:.2f} y={state[1]:.2f}")

print(f"\nAverage reward: {np.mean(total_rewards):.1f}")
print("Environment working correctly!")