import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Load training history ────────────────────────────────────────
history = np.load('models/training_history.npy', allow_pickle=True).item()
rewards  = history['rewards']
success  = history['success']
epsilon  = history['epsilon']
loss     = history['loss']

episodes = range(1, len(rewards) + 1)

# ── Smooth helper ────────────────────────────────────────────────
def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DQN Drone Landing — Training Analysis', fontsize=14, fontweight='bold')
fig.patch.set_facecolor('#0A0E14')
for ax in axes.flatten():
    ax.set_facecolor('#0F1520')
    ax.tick_params(colors='#7A9CC0')
    ax.xaxis.label.set_color('#7A9CC0')
    ax.yaxis.label.set_color('#7A9CC0')
    ax.title.set_color('#00D4FF')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E2D45')

# 1 — Reward curve
ax = axes[0, 0]
ax.plot(episodes, rewards, color='#1E2D45', alpha=0.3, linewidth=0.5)
ax.plot(range(50, len(rewards)+1), smooth(rewards),
        color='#00D4FF', linewidth=2, label='Smoothed')
ax.axhline(0, color='#81C784', linestyle='--', alpha=0.5, label='Break-even')
ax.set_title('Episode Reward')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.legend(facecolor='#141C2B', labelcolor='#7A9CC0')

# 2 — Success rate
ax = axes[0, 1]
success_smooth = smooth(success, window=50) * 100
ax.plot(range(50, len(success)+1), success_smooth,
        color='#81C784', linewidth=2)
ax.fill_between(range(50, len(success)+1), success_smooth,
                alpha=0.2, color='#81C784')
ax.set_title('Success Rate (%)')
ax.set_xlabel('Episode')
ax.set_ylabel('Success %')
ax.set_ylim(0, 100)

# 3 — Epsilon decay
ax = axes[1, 0]
ax.plot(episodes, epsilon, color='#FFB74D', linewidth=2)
ax.set_title('Exploration Rate (Epsilon)')
ax.set_xlabel('Episode')
ax.set_ylabel('Epsilon')

# 4 — Loss
ax = axes[1, 1]
loss_clean = [l for l in loss if not np.isnan(l) and l < 50]
ax.plot(range(len(loss_clean)), loss_clean,
        color='#F06292', linewidth=1, alpha=0.7)
ax.plot(range(50, len(loss_clean)+1),
        smooth(loss_clean),
        color='#CE93D8', linewidth=2)
ax.set_title('Training Loss')
ax.set_xlabel('Episode')
ax.set_ylabel('Huber Loss')

plt.tight_layout()
plt.savefig('docs/training_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0A0E14')
plt.show()
print("Saved to docs/training_analysis.png")