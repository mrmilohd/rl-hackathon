"""
BELLATRIX Report Generator
============================
Generates all figures, metrics, and analysis for the hackathon report.
Uses best_sac_agent.zip directly (requires stable-baselines3).

Outputs:
    report_figures/  — All PNG figures for the report
    report_metrics.txt — All numerical results in one place

Usage:
    python report_gen.py
    python report_gen.py --model best_sac_agent.zip
    python report_gen.py --episodes 200
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os
import time
import argparse
import sys

# Ensure Unicode-safe output on Windows terminals.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ============================================================
# ENVIRONMENT
# ============================================================
class MysteryControlEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        self.render_mode = render_mode
        self.state = None
        self.max_steps = 200
        self.current_step = 0
        self.inlet_flow_rate = 10.0
        self.outlet_flow_rate = 8.0
        self.heat_coefficient = 5.0
        self.cooling_coefficient = 2.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pressure = self.np_random.uniform(20, 40)
        temp = self.np_random.uniform(20, 30)
        target_pressure = self.np_random.uniform(50, 70)
        target_temp = self.np_random.uniform(60, 80)
        self.state = np.array([pressure, temp, target_pressure, target_temp], dtype=np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        inlet_v, outlet_v, heater_p = action
        pressure, temp, target_pressure, target_temp = self.state
        pressure_change = (inlet_v * self.inlet_flow_rate) - (outlet_v * self.outlet_flow_rate)
        new_pressure = np.clip(pressure + pressure_change, 0, 100)
        temp_change = (heater_p * self.heat_coefficient) - (self.cooling_coefficient * (temp / 100))
        new_temp = np.clip(temp + temp_change, 0, 100)
        terminated = bool(new_pressure > 95 or new_temp > 95)
        pressure_error = abs(new_pressure - target_pressure)
        temp_error = abs(new_temp - target_temp)
        reward = -(pressure_error + temp_error)
        if terminated:
            reward -= 100
        self.state = np.array([new_pressure, new_temp, target_pressure, target_temp], dtype=np.float32)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return self.state, reward, terminated, truncated, {}

    def close(self):
        pass


class NormalizedObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    def observation(self, obs):
        return obs / 100.0


# ============================================================
# STYLE CONFIG
# ============================================================
COLORS = {
    'pressure': '#2196F3',
    'temp': '#FF9800',
    'target': '#E53935',
    'score': '#4CAF50',
    'action_inlet': '#1976D2',
    'action_outlet': '#D32F2F',
    'action_heater': '#FF8F00',
    'safe_zone': '#C8E6C9',
    'danger_zone': '#FFCDD2',
    'bg': '#FAFAFA',
    'grid': '#E0E0E0',
    'accent': '#7C4DFF',
}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)


# ============================================================
# DATA COLLECTION
# ============================================================
def collect_episode(model, seed=42):
    """Run one episode, return full trace."""
    env = NormalizedObsWrapper(MysteryControlEnv())
    obs, _ = env.reset(seed=seed)
    raw = obs * 100.0

    trace = {
        'pressure': [raw[0]], 'temp': [raw[1]],
        'target_p': [raw[2]], 'target_t': [raw[3]],
        'p_error': [abs(raw[0]-raw[2])], 't_error': [abs(raw[1]-raw[3])],
        'inlet_v': [], 'outlet_v': [], 'heater_p': [],
        'cum_score': [0.0],
    }

    cum = 0.0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        raw = obs * 100.0

        pe = abs(raw[0] - raw[2])
        te = abs(raw[1] - raw[3])
        cum += -(pe + te)

        trace['pressure'].append(raw[0])
        trace['temp'].append(raw[1])
        trace['target_p'].append(raw[2])
        trace['target_t'].append(raw[3])
        trace['p_error'].append(pe)
        trace['t_error'].append(te)
        trace['inlet_v'].append(float(action[0]))
        trace['outlet_v'].append(float(action[1]))
        trace['heater_p'].append(float(action[2]))
        trace['cum_score'].append(cum)

        if terminated:
            break

    env.close()
    return trace


def collect_bulk(model, n_episodes=200):
    """Run many episodes, return summary data."""
    all_scores = []
    all_settle_2 = []
    all_settle_1 = []
    all_p_errors_by_step = []
    all_t_errors_by_step = []
    deaths = 0

    for seed in range(n_episodes):
        env = NormalizedObsWrapper(MysteryControlEnv())
        obs, _ = env.reset(seed=seed)
        score = 0.0
        dead = False
        s2 = s1 = None
        ep_p_err = []
        ep_t_err = []

        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = env.step(action)
            raw = obs * 100.0
            pe = abs(raw[0] - raw[2])
            te = abs(raw[1] - raw[3])
            score += -(pe + te)
            ep_p_err.append(pe)
            ep_t_err.append(te)

            if s2 is None and pe < 2.0 and te < 2.0:
                s2 = step
            if s1 is None and pe < 1.0 and te < 1.0:
                s1 = step

            if terminated:
                remaining = 200 - step - 1
                score += remaining * -(pe + te)
                dead = True
                # Pad remaining steps
                ep_p_err.extend([pe] * remaining)
                ep_t_err.extend([te] * remaining)
                break

        all_scores.append(score)
        if dead:
            deaths += 1
        if s2 is not None:
            all_settle_2.append(s2)
        if s1 is not None:
            all_settle_1.append(s1)
        all_p_errors_by_step.append(ep_p_err)
        all_t_errors_by_step.append(ep_t_err)
        env.close()

    return {
        'scores': np.array(all_scores),
        'deaths': deaths,
        'n': n_episodes,
        'settle_2': all_settle_2,
        'settle_1': all_settle_1,
        'p_errors_by_step': all_p_errors_by_step,
        't_errors_by_step': all_t_errors_by_step,
    }


# ============================================================
# FIGURE 1: Episode Trace (the hero figure)
# ============================================================
def fig_episode_trace(model, seed=42, save_dir="."):
    trace = collect_episode(model, seed)
    steps_state = range(len(trace['pressure']))
    steps_action = range(len(trace['inlet_v']))

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # Pressure tracking
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps_state, trace['pressure'], color=COLORS['pressure'], linewidth=2, label='Pressure')
    ax.plot(steps_state, trace['target_p'], color=COLORS['target'], linewidth=2, linestyle='--', label='Target')
    ax.axhline(95, color='darkred', linestyle=':', alpha=0.5, label='Safety Limit (95)')
    ax.fill_between(steps_state, 0, 95, alpha=0.03, color='green')
    ax.fill_between(steps_state, 95, 100, alpha=0.1, color='red')
    style_ax(ax, "Pressure Tracking", "Step", "Pressure (psi)")
    ax.legend(fontsize=9)

    # Temperature tracking
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps_state, trace['temp'], color=COLORS['temp'], linewidth=2, label='Temperature')
    ax.plot(steps_state, trace['target_t'], color=COLORS['target'], linewidth=2, linestyle='--', label='Target')
    ax.axhline(95, color='darkred', linestyle=':', alpha=0.5, label='Safety Limit (95)')
    ax.fill_between(steps_state, 0, 95, alpha=0.03, color='green')
    ax.fill_between(steps_state, 95, 100, alpha=0.1, color='red')
    style_ax(ax, "Temperature Tracking", "Step", "Temperature (°C)")
    ax.legend(fontsize=9)

    # Errors
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps_state, trace['p_error'], color=COLORS['pressure'], linewidth=1.5, alpha=0.8, label='Pressure Error')
    ax.plot(steps_state, trace['t_error'], color=COLORS['temp'], linewidth=1.5, alpha=0.8, label='Temp Error')
    total_err = [p + t for p, t in zip(trace['p_error'], trace['t_error'])]
    ax.plot(steps_state, total_err, color=COLORS['target'], linewidth=2, label='Total Error')
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.4, label='Settled Threshold (2.0)')
    style_ax(ax, "Tracking Errors Over Time", "Step", "Absolute Error")
    ax.legend(fontsize=9)

    # Cumulative score
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps_state, trace['cum_score'], color=COLORS['score'], linewidth=2.5)
    ax.axhline(-300, color='gray', linestyle='--', alpha=0.5, label='Hardcoded Best (~-300)')
    ax.fill_between(steps_state, trace['cum_score'], alpha=0.15, color=COLORS['score'])
    style_ax(ax, f"Cumulative Score (Final: {trace['cum_score'][-1]:.1f})", "Step", "Cumulative Score")
    ax.legend(fontsize=9)

    # Actions: Valves
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(steps_action, trace['inlet_v'], color=COLORS['action_inlet'], linewidth=1.2, alpha=0.8, label='Inlet Valve')
    ax.plot(steps_action, trace['outlet_v'], color=COLORS['action_outlet'], linewidth=1.2, alpha=0.8, label='Outlet Valve')
    ax.set_ylim(-0.05, 1.05)
    style_ax(ax, "Valve Control Actions", "Step", "Valve Opening [0-1]")
    ax.legend(fontsize=9)

    # Actions: Heater
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(steps_action, trace['heater_p'], color=COLORS['action_heater'], linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)
    # Mark equilibrium heater for target temp
    if trace['target_t']:
        eq_heater = 2.0 * (trace['target_t'][-1] / 100.0) / 5.0
        ax.axhline(eq_heater, color='gray', linestyle='--', alpha=0.5,
                    label=f'Equilibrium ({eq_heater:.3f})')
    style_ax(ax, "Heater Power Control", "Step", "Heater Power [0-1]")
    ax.legend(fontsize=9)

    fig.suptitle(f"SAC Agent — Episode Trace (seed={seed})", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(f"{save_dir}/fig1_episode_trace.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig1_episode_trace.png")


# ============================================================
# FIGURE 2: Score Distribution
# ============================================================
def fig_score_distribution(bulk, save_dir="."):
    scores = bulk['scores']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    ax = axes[0]
    n, bins, patches = ax.hist(scores, bins=35, color=COLORS['accent'], edgecolor='white', alpha=0.85)
    ax.axvline(scores.mean(), color=COLORS['target'], linestyle='--', linewidth=2,
               label=f'Mean: {scores.mean():.1f}')
    ax.axvline(-300, color='gray', linestyle=':', linewidth=2, label='Hardcoded Best: -300')
    style_ax(ax, "Score Distribution", "Episode Score", "Count")
    ax.legend(fontsize=9)

    # CDF
    ax = axes[1]
    sorted_s = np.sort(scores)
    cdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s) * 100
    ax.plot(sorted_s, cdf, color=COLORS['accent'], linewidth=2.5)
    ax.axvline(-300, color='gray', linestyle=':', linewidth=2, label='Hardcoded Best')
    ax.axhline(50, color=COLORS['grid'], linestyle='--', alpha=0.5)
    ax.fill_betweenx(cdf, sorted_s, alpha=0.1, color=COLORS['accent'])
    style_ax(ax, "Cumulative Distribution", "Episode Score", "Percentile (%)")
    ax.legend(fontsize=9)

    # Box plot
    ax = axes[2]
    bp = ax.boxplot(scores, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=COLORS['accent'], alpha=0.5),
                    medianprops=dict(color=COLORS['target'], linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    ax.axhline(-300, color='gray', linestyle=':', linewidth=2, label='Hardcoded Best')
    ax.set_xticklabels(['SAC Agent'])
    style_ax(ax, "Score Box Plot", "", "Episode Score")
    ax.legend(fontsize=9)

    plt.suptitle(f"Score Analysis ({bulk['n']} episodes)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig2_score_distribution.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig2_score_distribution.png")


# ============================================================
# FIGURE 3: Error Decay Heatmap (all episodes stacked)
# ============================================================
def fig_error_heatmap(bulk, save_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Pad all episodes to 200 steps
    def pad_to_200(arr_list):
        padded = []
        for arr in arr_list:
            if len(arr) < 200:
                arr = arr + [arr[-1]] * (200 - len(arr))
            padded.append(arr[:200])
        return np.array(padded)

    p_matrix = pad_to_200(bulk['p_errors_by_step'])
    t_matrix = pad_to_200(bulk['t_errors_by_step'])

    # Sort by final error for visual clarity
    sort_idx = np.argsort(p_matrix[:, -1])

    cmap = LinearSegmentedColormap.from_list('custom', ['#1B5E20', '#FDD835', '#E53935'])

    ax = axes[0]
    im = ax.imshow(p_matrix[sort_idx], aspect='auto', cmap=cmap, vmin=0, vmax=20,
                   extent=[0, 200, 0, len(p_matrix)])
    style_ax(ax, "Pressure Error by Episode & Step", "Step", "Episode (sorted)")
    plt.colorbar(im, ax=ax, label='|P - Target_P|')

    sort_idx_t = np.argsort(t_matrix[:, -1])
    ax = axes[1]
    im = ax.imshow(t_matrix[sort_idx_t], aspect='auto', cmap=cmap, vmin=0, vmax=20,
                   extent=[0, 200, 0, len(t_matrix)])
    style_ax(ax, "Temperature Error by Episode & Step", "Step", "Episode (sorted)")
    plt.colorbar(im, ax=ax, label='|T - Target_T|')

    plt.suptitle("Error Decay Heatmap — All Episodes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig3_error_heatmap.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig3_error_heatmap.png")


# ============================================================
# FIGURE 4: Mean Error Decay Curve with Confidence Band
# ============================================================
def fig_mean_error_curve(bulk, save_dir="."):
    def pad_to_200(arr_list):
        padded = []
        for arr in arr_list:
            if len(arr) < 200:
                arr = arr + [arr[-1]] * (200 - len(arr))
            padded.append(arr[:200])
        return np.array(padded)

    p_matrix = pad_to_200(bulk['p_errors_by_step'])
    t_matrix = pad_to_200(bulk['t_errors_by_step'])
    total_matrix = p_matrix + t_matrix

    steps = np.arange(200)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Per-channel
    ax = axes[0]
    for matrix, label, color in [
        (p_matrix, 'Pressure Error', COLORS['pressure']),
        (t_matrix, 'Temp Error', COLORS['temp']),
    ]:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        p25 = np.percentile(matrix, 25, axis=0)
        p75 = np.percentile(matrix, 75, axis=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=f'{label} (mean)')
        ax.fill_between(steps, p25, p75, alpha=0.2, color=color)
    ax.axhline(2.0, color='gray', linestyle='--', alpha=0.4, label='Settled (2.0)')
    style_ax(ax, "Per-Channel Error Decay", "Step", "Absolute Error")
    ax.legend(fontsize=9)
    ax.set_ylim(0, None)

    # Total error
    ax = axes[1]
    mean = total_matrix.mean(axis=0)
    std = total_matrix.std(axis=0)
    p10 = np.percentile(total_matrix, 10, axis=0)
    p90 = np.percentile(total_matrix, 90, axis=0)
    ax.plot(steps, mean, color=COLORS['accent'], linewidth=2.5, label='Mean Total Error')
    ax.fill_between(steps, p10, p90, alpha=0.15, color=COLORS['accent'], label='10th–90th percentile')
    ax.axhline(4.0, color='gray', linestyle='--', alpha=0.4, label='Combined Threshold (4.0)')
    style_ax(ax, "Total Error Decay (P + T)", "Step", "Total Absolute Error")
    ax.legend(fontsize=9)
    ax.set_ylim(0, None)

    plt.suptitle(f"Error Convergence Analysis ({bulk['n']} episodes)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig4_error_decay.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig4_error_decay.png")


# ============================================================
# FIGURE 5: Phase Portrait (Pressure vs Temperature trajectory)
# ============================================================
def fig_phase_portrait(model, save_dir="."):
    fig, ax = plt.subplots(figsize=(10, 9))

    for seed in range(20):
        trace = collect_episode(model, seed)
        p = trace['pressure']
        t = trace['temp']
        tp = trace['target_p'][0]
        tt = trace['target_t'][0]

        # Color by time (lighter = later)
        n_pts = len(p)
        colors = plt.cm.viridis(np.linspace(0.2, 1.0, n_pts))
        for i in range(n_pts - 1):
            ax.plot([p[i], p[i+1]], [t[i], t[i+1]], color=colors[i],
                    linewidth=0.8, alpha=0.6)

        # Start and target markers
        ax.scatter(p[0], t[0], color='blue', s=30, zorder=5, marker='o', alpha=0.5)
        ax.scatter(tp, tt, color='red', s=50, zorder=5, marker='x', linewidths=2, alpha=0.5)

    # Safety boundary
    ax.axvline(95, color='darkred', linestyle=':', alpha=0.3)
    ax.axhline(95, color='darkred', linestyle=':', alpha=0.3)
    ax.fill_between([95, 100], 0, 100, alpha=0.05, color='red')
    ax.fill_betweenx([95, 100], 0, 100, alpha=0.05, color='red')

    # Start and target regions
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((20, 20), 20, 10, fill=False, edgecolor='blue',
                            linestyle='--', linewidth=1.5, label='Start Region'))
    ax.add_patch(Rectangle((50, 60), 20, 20, fill=False, edgecolor='red',
                            linestyle='--', linewidth=1.5, label='Target Region'))

    style_ax(ax, "Phase Portrait: Pressure vs Temperature Trajectories",
             "Pressure (psi)", "Temperature (°C)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper left')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 200))
    plt.colorbar(sm, ax=ax, label='Timestep', shrink=0.7)

    plt.savefig(f"{save_dir}/fig5_phase_portrait.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig5_phase_portrait.png")


# ============================================================
# FIGURE 6: Settling Time Histogram
# ============================================================
def fig_settling_time(bulk, save_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, threshold, color in [
        (axes[0], bulk['settle_2'], 2.0, COLORS['pressure']),
        (axes[1], bulk['settle_1'], 1.0, COLORS['temp']),
    ]:
        if data:
            ax.hist(data, bins=range(0, max(data)+2), color=color, edgecolor='white', alpha=0.8)
            ax.axvline(np.mean(data), color=COLORS['target'], linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(data):.1f} steps')
            ax.axvline(np.median(data), color='gray', linestyle=':', linewidth=2,
                       label=f'Median: {np.median(data):.0f} steps')
            settled_pct = len(data) / bulk['n'] * 100
            style_ax(ax, f"Settling Time (error < {threshold}) — {settled_pct:.0f}% settled",
                     "Steps to Settle", "Count")
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, f"No episodes settled\nbelow {threshold}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            style_ax(ax, f"Settling Time (error < {threshold})", "Steps", "Count")

    plt.suptitle("Settling Time Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig6_settling_time.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig6_settling_time.png")


# ============================================================
# FIGURE 7: Action Distribution Analysis
# ============================================================
def fig_action_distribution(model, save_dir="."):
    all_inlet = []
    all_outlet = []
    all_heater = []
    phases = {'ramp': [], 'settle': [], 'hold': []}  # step < 15, 15-30, 30+

    for seed in range(100):
        trace = collect_episode(model, seed)
        for i, (iv, ov, hp) in enumerate(zip(trace['inlet_v'], trace['outlet_v'], trace['heater_p'])):
            all_inlet.append(iv)
            all_outlet.append(ov)
            all_heater.append(hp)
            if i < 15:
                phases['ramp'].append([iv, ov, hp])
            elif i < 30:
                phases['settle'].append([iv, ov, hp])
            else:
                phases['hold'].append([iv, ov, hp])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: overall histograms
    for ax, data, label, color in [
        (axes[0, 0], all_inlet, 'Inlet Valve', COLORS['action_inlet']),
        (axes[0, 1], all_outlet, 'Outlet Valve', COLORS['action_outlet']),
        (axes[0, 2], all_heater, 'Heater Power', COLORS['action_heater']),
    ]:
        ax.hist(data, bins=50, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(data), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean: {np.mean(data):.3f}')
        style_ax(ax, f'{label} Distribution', 'Value [0-1]', 'Count')
        ax.legend(fontsize=9)

    # Bottom row: by phase
    phase_names = ['ramp', 'settle', 'hold']
    phase_labels = ['Ramp-up (0-15)', 'Settling (15-30)', 'Holding (30+)']
    action_names = ['Inlet', 'Outlet', 'Heater']
    action_colors = [COLORS['action_inlet'], COLORS['action_outlet'], COLORS['action_heater']]

    for idx, (phase, plabel) in enumerate(zip(phase_names, phase_labels)):
        ax = axes[1, idx]
        if phases[phase]:
            data = np.array(phases[phase])
            positions = [1, 2, 3]
            bp = ax.boxplot([data[:, 0], data[:, 1], data[:, 2]],
                           positions=positions, patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], action_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax.set_xticklabels(action_names)
            ax.set_ylim(-0.05, 1.05)
        style_ax(ax, f'Actions: {plabel}', '', 'Value [0-1]')

    plt.suptitle("Action Distribution Analysis (100 episodes)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig7_action_distribution.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig7_action_distribution.png")


# ============================================================
# FIGURE 8: Robustness (perturbed dynamics)
# ============================================================
def fig_robustness(model, save_dir="."):
    perturbations = [
        ("Baseline", 1.0, 1.0, 1.0, 1.0),
        ("Inlet+20%", 1.2, 1.0, 1.0, 1.0),
        ("Inlet−20%", 0.8, 1.0, 1.0, 1.0),
        ("Outlet+20%", 1.0, 1.2, 1.0, 1.0),
        ("Outlet−20%", 1.0, 0.8, 1.0, 1.0),
        ("Heat+20%", 1.0, 1.0, 1.2, 1.0),
        ("Heat−20%", 1.0, 1.0, 0.8, 1.0),
        ("Cool+20%", 1.0, 1.0, 1.0, 1.2),
        ("Cool−20%", 1.0, 1.0, 1.0, 0.8),
        ("All+20%", 1.2, 1.2, 1.2, 1.2),
        ("All−20%", 0.8, 0.8, 0.8, 0.8),
    ]

    names = []
    means = []
    stds = []
    comp_rates = []

    for name, im, om, hm, cm in perturbations:
        ep_scores = []
        alive = 0
        for seed in range(50):
            env = MysteryControlEnv()
            env.inlet_flow_rate = 10.0 * im
            env.outlet_flow_rate = 8.0 * om
            env.heat_coefficient = 5.0 * hm
            env.cooling_coefficient = 2.0 * cm
            wrapped = NormalizedObsWrapper(env)
            obs, _ = wrapped.reset(seed=seed)
            score = 0.0
            dead = False
            for step in range(200):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, _, _ = wrapped.step(action)
                raw = obs * 100.0
                pe = abs(raw[0] - raw[2])
                te = abs(raw[1] - raw[3])
                score += -(pe + te)
                if terminated:
                    remaining = 200 - step - 1
                    score += remaining * -(pe + te)
                    dead = True
                    break
            ep_scores.append(score)
            if not dead:
                alive += 1
            wrapped.close()

        names.append(name)
        means.append(np.mean(ep_scores))
        stds.append(np.std(ep_scores))
        comp_rates.append(alive / 50 * 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    x = np.arange(len(names))
    colors = [COLORS['score'] if cr == 100 else COLORS['temp'] if cr > 80 else COLORS['target']
              for cr in comp_rates]

    ax1.bar(x, means, yerr=stds, color=colors, edgecolor='white', alpha=0.85, capsize=4)
    ax1.axhline(means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    style_ax(ax1, "Mean Score Under Perturbed Dynamics", "", "Mean Score")
    ax1.legend(fontsize=9)

    ax2.bar(x, comp_rates, color=colors, edgecolor='white', alpha=0.85)
    ax2.axhline(100, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 110)
    style_ax(ax2, "Completion Rate Under Perturbed Dynamics", "", "Completion %")

    plt.suptitle("Robustness Analysis: ±20% Parameter Perturbation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig8_robustness.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig8_robustness.png")


# ============================================================
# FIGURE 9: Multi-Episode Overlay (5 diverse seeds)
# ============================================================
def fig_multi_episode(model, save_dir="."):
    seeds = [0, 17, 42, 73, 99]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for seed in seeds:
        trace = collect_episode(model, seed)
        steps = range(len(trace['pressure']))
        alpha = 0.7

        axes[0, 0].plot(steps, trace['pressure'], linewidth=1.2, alpha=alpha, label=f'Seed {seed}')
        axes[0, 1].plot(steps, trace['temp'], linewidth=1.2, alpha=alpha, label=f'Seed {seed}')
        axes[1, 0].plot(steps, trace['p_error'], linewidth=1, alpha=alpha)
        axes[1, 0].plot(steps, trace['t_error'], linewidth=1, alpha=alpha, linestyle='--')
        axes[1, 1].plot(steps, trace['cum_score'], linewidth=1.5, alpha=alpha, label=f'Seed {seed}')

    for trace_seed in seeds:
        trace = collect_episode(model, trace_seed)
        tp, tt = trace['target_p'][0], trace['target_t'][0]
        axes[0, 0].axhline(tp, color='gray', linestyle=':', alpha=0.15)
        axes[0, 1].axhline(tt, color='gray', linestyle=':', alpha=0.15)

    style_ax(axes[0, 0], "Pressure Across Seeds", "Step", "Pressure (psi)")
    style_ax(axes[0, 1], "Temperature Across Seeds", "Step", "Temperature (°C)")
    style_ax(axes[1, 0], "Errors (solid=P, dashed=T)", "Step", "Absolute Error")
    style_ax(axes[1, 1], "Cumulative Score", "Step", "Score")

    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].axhline(-300, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle("Multi-Episode Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig9_multi_episode.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig9_multi_episode.png")


# ============================================================
# FIGURE 10: Score vs Initial Gap
# ============================================================
def fig_score_vs_gap(model, save_dir="."):
    p_gaps = []
    t_gaps = []
    total_gaps = []
    scores = []

    for seed in range(200):
        env = MysteryControlEnv()
        obs, _ = env.reset(seed=seed)
        p, t, tp, tt = obs
        p_gap = abs(tp - p)
        t_gap = abs(tt - t)
        p_gaps.append(p_gap)
        t_gaps.append(t_gap)
        total_gaps.append(p_gap + t_gap)

        wrapped = NormalizedObsWrapper(env)
        obs_n = wrapped.observation(obs)
        score = 0.0
        for step in range(200):
            action, _ = model.predict(obs_n, deterministic=True)
            obs_n, _, terminated, _, _ = wrapped.step(action)
            raw = obs_n * 100.0
            score += -(abs(raw[0] - raw[2]) + abs(raw[1] - raw[3]))
            if terminated:
                remaining = 200 - step - 1
                raw2 = obs_n * 100.0
                score += remaining * -(abs(raw2[0] - raw2[2]) + abs(raw2[1] - raw2[3]))
                break
        scores.append(score)
        wrapped.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, gaps, label, color in [
        (axes[0], p_gaps, 'Pressure Gap', COLORS['pressure']),
        (axes[1], t_gaps, 'Temperature Gap', COLORS['temp']),
        (axes[2], total_gaps, 'Total Gap', COLORS['accent']),
    ]:
        ax.scatter(gaps, scores, c=color, alpha=0.5, s=25, edgecolors='white', linewidth=0.3)
        # Trend line
        z = np.polyfit(gaps, scores, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(gaps), max(gaps), 50)
        ax.plot(x_line, p_line(x_line), color='black', linewidth=2, linestyle='--',
                label=f'Trend (slope={z[0]:.1f})')
        corr = np.corrcoef(gaps, scores)[0, 1]
        style_ax(ax, f'Score vs {label} (r={corr:.2f})', f'Initial {label}', 'Episode Score')
        ax.legend(fontsize=9)

    plt.suptitle("Impact of Initial Conditions on Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig10_score_vs_gap.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig10_score_vs_gap.png")


# ============================================================
# FIGURE 11: Reward Function Landscape Visualization
# ============================================================
def fig_reward_landscape(save_dir="."):
    """Visualize the shaped reward function as a 2D surface."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reward as function of (p_error, t_error)
    p_err = np.linspace(0, 30, 100)
    t_err = np.linspace(0, 30, 100)
    P, T = np.meshgrid(p_err, t_err)

    # Shaped reward (simplified, without shaping term)
    R = -(P + T)
    R = np.where(P < 2.0, R + 3.0, R)
    R = np.where(T < 2.0, R + 3.0, R)
    R = np.where((P < 0.5) & (T < 0.5), R + 10.0, R)

    ax = axes[0]
    im = ax.contourf(P, T, R, levels=30, cmap='RdYlGn')
    ax.contour(P, T, R, levels=10, colors='black', linewidths=0.5, alpha=0.3)
    ax.plot(0, 0, 'w*', markersize=15, markeredgecolor='black', label='Optimal (0, 0)')
    plt.colorbar(im, ax=ax, label='Shaped Reward')
    style_ax(ax, "Reward Landscape", "Pressure Error", "Temperature Error")
    ax.legend(fontsize=10)

    # Safety penalty overlay
    vals = np.linspace(0, 100, 200)
    P2, T2 = np.meshgrid(vals, vals)
    penalty = np.zeros_like(P2)
    penalty = np.where(P2 > 85, penalty - 5.0 * (P2 - 85), penalty)
    penalty = np.where(T2 > 85, penalty - 5.0 * (T2 - 85), penalty)
    penalty = np.where((P2 > 95) | (T2 > 95), penalty - 200, penalty)

    ax = axes[1]
    im = ax.contourf(P2, T2, penalty, levels=30, cmap='RdYlGn_r')
    ax.axvline(95, color='darkred', linewidth=2, label='Kill Zone (95)')
    ax.axhline(95, color='darkred', linewidth=2)
    ax.axvline(85, color='orange', linestyle='--', linewidth=1.5, label='Soft Wall (85)')
    ax.axhline(85, color='orange', linestyle='--', linewidth=1.5)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((50, 60), 20, 20, fill=False, edgecolor='white',
                            linewidth=2, linestyle='--', label='Target Region'))
    plt.colorbar(im, ax=ax, label='Safety Penalty')
    style_ax(ax, "Safety Penalty Map", "Pressure", "Temperature")
    ax.legend(fontsize=9, loc='upper left')

    plt.suptitle("Reward Function Design", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fig11_reward_landscape.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ fig11_reward_landscape.png")


# ============================================================
# METRICS SUMMARY FILE
# ============================================================
def write_metrics(bulk, save_dir="."):
    scores = bulk['scores']
    lines = []
    lines.append("=" * 60)
    lines.append("BELLATRIX — FINAL METRICS SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Episodes evaluated:   {bulk['n']}")
    lines.append(f"Completion rate:      {(bulk['n'] - bulk['deaths'])}/{bulk['n']} "
                 f"({(bulk['n'] - bulk['deaths'])/bulk['n']*100:.1f}%)")
    lines.append("")
    lines.append("--- Score Statistics ---")
    lines.append(f"Mean:                 {scores.mean():.2f}")
    lines.append(f"Std:                  {scores.std():.2f}")
    lines.append(f"Median:               {np.median(scores):.2f}")
    lines.append(f"Min (worst):          {scores.min():.2f}")
    lines.append(f"Max (best):           {scores.max():.2f}")
    lines.append(f"5th percentile:       {np.percentile(scores, 5):.2f}")
    lines.append(f"25th percentile:      {np.percentile(scores, 25):.2f}")
    lines.append(f"75th percentile:      {np.percentile(scores, 75):.2f}")
    lines.append(f"95th percentile:      {np.percentile(scores, 95):.2f}")
    lines.append(f"Hardcoded best:       ~-300")
    lines.append("")
    lines.append("--- Settling Time ---")
    if bulk['settle_2']:
        lines.append(f"Error < 2.0:  mean={np.mean(bulk['settle_2']):.1f}  "
                     f"median={np.median(bulk['settle_2']):.0f}  "
                     f"({len(bulk['settle_2'])}/{bulk['n']} episodes)")
    if bulk['settle_1']:
        lines.append(f"Error < 1.0:  mean={np.mean(bulk['settle_1']):.1f}  "
                     f"median={np.median(bulk['settle_1']):.0f}  "
                     f"({len(bulk['settle_1'])}/{bulk['n']} episodes)")
    lines.append("")
    lines.append("--- Environment Parameters ---")
    lines.append(f"Inlet flow rate:      10.0")
    lines.append(f"Outlet flow rate:     8.0")
    lines.append(f"Heat coefficient:     5.0")
    lines.append(f"Cooling coefficient:  2.0")
    lines.append(f"Max steps:            200")
    lines.append(f"Safety limit:         95 (pressure & temp)")
    lines.append("")
    lines.append("--- Agent Architecture ---")
    lines.append(f"Algorithm:            SAC (Soft Actor-Critic)")
    lines.append(f"Network:              MLP [4 → 256 → 256 → 3]")
    lines.append(f"Activation:           ReLU (hidden), Tanh (output)")
    lines.append(f"Output scaling:       Tanh → [0, 1]")
    lines.append(f"Observation norm:     /100.0")
    lines.append("")
    lines.append("--- Hyperparameters ---")
    lines.append(f"Learning rate:        3e-4")
    lines.append(f"Replay buffer:        200,000")
    lines.append(f"Batch size:           256")
    lines.append(f"Gamma:                0.99")
    lines.append(f"Tau:                  0.005")
    lines.append(f"Entropy:              auto")
    lines.append("=" * 60)

    text = "\n".join(lines)
    with open(f"{save_dir}/report_metrics.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n{text}")
    print(f"\n  ✓ report_metrics.txt saved")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="BELLATRIX Report Generator")
    parser.add_argument("--model", default="best_sac_agent", help="Path to SB3 model")
    parser.add_argument("--episodes", type=int, default=200, help="Bulk eval episodes")
    parser.add_argument("--outdir", default="report_figures", help="Output directory")
    args = parser.parse_args()

    from stable_baselines3 import SAC

    save_dir = args.outdir
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("BELLATRIX REPORT GENERATOR")
    print("=" * 60)

    # Load model
    model_path = args.model
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    model = SAC.load(model_path)
    print(f"Loaded: {model_path}")
    print(f"Output: {save_dir}/")
    print()

    start = time.time()

    # Collect bulk data
    print(f"Collecting bulk data ({args.episodes} episodes)...")
    bulk = collect_bulk(model, args.episodes)
    print(f"  Done. Mean score: {bulk['scores'].mean():.1f}\n")

    # Generate all figures
    print("Generating figures...")
    fig_episode_trace(model, seed=42, save_dir=save_dir)
    fig_score_distribution(bulk, save_dir=save_dir)
    fig_error_heatmap(bulk, save_dir=save_dir)
    fig_mean_error_curve(bulk, save_dir=save_dir)
    fig_phase_portrait(model, save_dir=save_dir)
    fig_settling_time(bulk, save_dir=save_dir)
    fig_action_distribution(model, save_dir=save_dir)
    fig_robustness(model, save_dir=save_dir)
    fig_multi_episode(model, save_dir=save_dir)
    fig_score_vs_gap(model, save_dir=save_dir)
    fig_reward_landscape(save_dir=save_dir)

    # Write metrics
    write_metrics(bulk, save_dir=save_dir)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL DONE — {elapsed:.0f}s")
    print(f"{'='*60}")
    print(f"Figures saved to: {save_dir}/")
    print(f"""
Files generated:
  fig1_episode_trace.png        — Hero figure: full episode walkthrough
  fig2_score_distribution.png   — Histogram, CDF, box plot of scores
  fig3_error_heatmap.png        — Error decay across all episodes
  fig4_error_decay.png          — Mean error convergence with bands
  fig5_phase_portrait.png       — P vs T trajectory (20 episodes)
  fig6_settling_time.png        — How fast agent reaches target
  fig7_action_distribution.png  — What actions the agent takes
  fig8_robustness.png           — Performance under ±20% dynamics
  fig9_multi_episode.png        — 5 diverse episodes overlaid
  fig10_score_vs_gap.png        — Score correlation with initial gap
  fig11_reward_landscape.png    — Reward function visualization
  report_metrics.txt            — All numerical metrics
""")


if __name__ == "__main__":
    main()