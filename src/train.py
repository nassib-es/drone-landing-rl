import sys
import numpy as np
import os
sys.path.append('.')

from env.drone_env import DroneEnv
from src.dqn import DQNAgent

def train(episodes=1000, save_every=100):

    env   = DroneEnv()
    agent = DQNAgent(
        state_size    = env.state_size,
        action_size   = env.action_size,
        lr            = 0.001,
        gamma         = 0.99,
        epsilon       = 1.0,
        epsilon_min   = 0.01,
        epsilon_decay = 0.997,
        batch_size    = 64,
        target_update = 100
    )

    # Tracking
    rewards_history  = []
    success_history  = []
    epsilon_history  = []
    loss_history     = []

    best_reward = -np.inf

    print("=" * 60)
    print("Training DQN Drone Landing Agent")
    print(f"Episodes: {episodes} | Batch: {agent.batch_size}")
    print(f"Gamma: {agent.gamma} | LR: {agent.optimizer.param_groups[0]['lr']}")
    print("=" * 60)

    for episode in range(1, episodes + 1):
        state       = env.reset()
        total_reward = 0
        done        = False
        losses      = []
        landed      = False

        while not done:
            action                    = agent.act(state)
            next_state, reward, done  = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state        = next_state
            total_reward += reward

            # Check if landed successfully
            if done and reward >= env.REWARD_LAND * 0.9:
                landed = True

        rewards_history.append(total_reward)
        success_history.append(1 if landed else 0)
        epsilon_history.append(agent.epsilon)
        loss_history.append(np.mean(losses) if losses else 0)

        # Decay epsilon per episode not per step
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            os.makedirs('models', exist_ok=True)
            agent.save('models/best_model.pt')

        # Save checkpoint
        if episode % save_every == 0:
            agent.save(f'models/checkpoint_{episode}.pt')

        # Print progress
        if episode % 50 == 0:
            avg_reward  = np.mean(rewards_history[-50:])
            success_rate = np.mean(success_history[-50:]) * 100
            avg_loss    = np.mean(loss_history[-50:])
            print(f"Ep {episode:5d} | "
                  f"Avg Reward: {avg_reward:8.1f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

    print("\nTraining complete!")
    print(f"Best reward: {best_reward:.1f}")

    # Save training history
    os.makedirs('models', exist_ok=True)
    np.save('models/training_history.npy', {
        'rewards':  rewards_history,
        'success':  success_history,
        'epsilon':  epsilon_history,
        'loss':     loss_history
    }, allow_pickle=True)

    return agent, rewards_history, success_history

if __name__ == '__main__':
    agent, rewards, successes = train(episodes=3000)