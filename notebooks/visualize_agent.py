import sys
sys.path.append('.')
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from env.drone_env import DroneEnv
from src.dqn import DQNAgent

# ── Load best model ──────────────────────────────────────────────
env   = DroneEnv(platform_speed=1.5)
agent = DQNAgent(env.state_size, env.action_size)
agent.load('models/best_model.pt')
agent.epsilon = 0.0

# ── Run one episode ──────────────────────────────────────────────
state = env.reset()
frames = []
done   = False

while not done:
    frames.append(env.state.copy())
    action = agent.act(state)
    state, reward, done = env.step(action)
frames.append(env.state.copy())

print(f"Episode length: {len(frames)} steps")
print(f"Outcome: {'LANDED!' if reward > 50 else 'CRASHED'}")

# ── Setup figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#0A0E14')
ax.set_facecolor('#0F1520')
ax.set_xlim(-10.0, 10.0)
ax.set_ylim(-0.5, 15.0)
ax.tick_params(colors='#7A9CC0')
for spine in ax.spines.values():
    spine.set_edgecolor('#1E2D45')
ax.set_title('DQN Drone Landing Agent', color='#00D4FF', fontsize=13)
ax.set_xlabel('X Position (m)', color='#7A9CC0')
ax.set_ylabel('Y Position (m)', color='#7A9CC0')

# Ground
ground = patches.Rectangle((-10.0, -0.5), 20.0, 0.5, color='#1E2D45')
ax.add_patch(ground)

# Platform
pw = env.PLATFORM_WIDTH
py = env.PLATFORM_Y

platform_patch = patches.Rectangle(
    (frames[0][4] - pw/2, py - 0.1),
    pw, 0.2, color='#00D4FF', zorder=3
)
ax.add_patch(platform_patch)

# Drone starting position
x0, y0 = frames[0][0], frames[0][1]

# Drone — rectangle body + 4 rotor arms
drone_body  = patches.FancyBboxPatch((x0-0.4, y0-0.15), 0.8, 0.3,
                boxstyle="round,pad=0.05", color='#FFB74D', zorder=4)
arm_l = patches.FancyArrow(x0-0.4, y0, -0.3, 0, width=0.05,
                color='#7A9CC0', zorder=3)
arm_r = patches.FancyArrow(x0+0.4, y0,  0.3, 0, width=0.05,
                color='#7A9CC0', zorder=3)
prop_l = plt.Circle((x0-0.7, y0), 0.18, color='#F06292', zorder=5, alpha=0.7)
prop_r = plt.Circle((x0+0.7, y0), 0.18, color='#F06292', zorder=5, alpha=0.7)
ax.add_patch(drone_body)
ax.add_patch(arm_l)
ax.add_patch(arm_r)
ax.add_patch(prop_l)
ax.add_patch(prop_r)

# Text
step_text    = ax.text(-9.5, 14.5, '', color='#7A9CC0', fontsize=9)
outcome_text = ax.text(0, 14.5, '', color='#81C784', fontsize=10,
                       ha='center', fontweight='bold')

def update(frame_idx):
    s = frames[frame_idx]
    x, y, vx, vy, px, pv = s

    drone_body.set_x(x - 0.4)
    drone_body.set_y(y - 0.15)
    arm_l.set_data(x=x-0.4, y=y, dx=-0.3, dy=0)
    arm_r.set_data(x=x+0.4, y=y, dx=0.3,  dy=0)
    prop_l.center = (x-0.7, y)
    prop_r.center = (x+0.7, y)
    platform_patch.set_x(px - pw/2)

    step_text.set_text(
        f'Step {frame_idx:3d} | '
        f'Drone ({x:.1f}, {y:.1f}) | '
        f'Vel ({vx:.1f}, {vy:.1f}) | '
        f'Platform {px:.1f}'
    )

    if frame_idx == len(frames) - 1:
        outcome_text.set_text('LANDED! ✓' if reward > 50 else 'CRASHED ✗')
        outcome_text.set_color('#81C784' if reward > 50 else '#FF4444')

    return drone_body, arm_l, arm_r, prop_l, prop_r, platform_patch, step_text, outcome_text

anim = animation.FuncAnimation(
    fig, update, frames=len(frames),
    interval=50, blit=True
)

print("Saving GIF...")
anim.save('docs/landing_demo.gif', writer='pillow', fps=20,
          savefig_kwargs={'facecolor': '#0A0E14'})
print("Saved to docs/landing_demo.gif")
plt.show()