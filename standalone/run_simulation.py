import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import time as _time

from standalone.world import World
from standalone.sensor import Sensor
from standalone.planner import MultiObjectivePlanner

# ─── Colour palette ──────────────────────────────────────────────────────────
CLR_BG       = '#1a1a2e'
CLR_ROAD     = '#16213e'
CLR_LANE     = '#e2b714'
CLR_EGO      = '#00d4ff'
CLR_AGENT    = '#ff8c00'
CLR_PED      = '#ff4757'
CLR_PLAN     = '#7bed9f'
CLR_PANEL    = '#0f3460'
CLR_TEXT     = '#e0e0e0'
CLR_OK       = '#2ecc71'
CLR_WARN     = '#f39c12'
CLR_DANGER   = '#e74c3c'


def signal_color(state):
    return {'GREEN': CLR_OK, 'YELLOW': CLR_WARN, 'RED': CLR_DANGER}.get(state, 'white')


def main():
    world   = World()
    sensor  = Sensor(range=70.0, noise_std=0.1)
    planner = MultiObjectivePlanner(target_speed=15.0)

    sim_start_real = _time.time()   # wall-clock start

    # ── Figure layout: main view (top) + dashboard (bottom) ──────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=CLR_BG)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.04)

    ax_main  = fig.add_subplot(gs[0])  # driving view
    ax_dash  = fig.add_subplot(gs[1])  # dashboard panel

    ax_main.set_facecolor(CLR_ROAD)
    ax_dash.set_facecolor(CLR_PANEL)
    ax_dash.axis('off')

    # Pre-create text objects in the dashboard for smooth animation
    dash_texts = {}
    text_cfg   = dict(fontsize=11, color=CLR_TEXT, fontfamily='monospace',
                      verticalalignment='top', transform=ax_dash.transAxes)

    # Column positions (in axes fraction 0-1)
    cols = [0.01, 0.22, 0.44, 0.66, 0.84]
    labels = ['TIME', 'DISTANCE', 'SPEED', 'VIOLATIONS', 'LAW STATUS']
    for i, (cx, lbl) in enumerate(zip(cols, labels)):
        ax_dash.text(cx, 0.95, lbl, fontsize=9, color=CLR_LANE, fontfamily='monospace',
                     fontweight='bold', transform=ax_dash.transAxes, va='top')
        dash_texts[i] = ax_dash.text(cx, 0.60, '', **text_cfg)

    # Violation log (last 4 lines, bottom of dashboard)
    vlog_texts = []
    for row in range(4):
        t = ax_dash.text(0.01, 0.42 - row * 0.12, '', fontsize=8.5,
                         color=CLR_DANGER, fontfamily='monospace',
                         transform=ax_dash.transAxes, va='top')
        vlog_texts.append(t)

    def animate(frame):
        # ── Perception ───────────────────────────────────────────────────────
        observations = sensor.get_observations(
            world.ego, world.agents, world.traffic_lights,
            world.stop_signs, world.speed_zones
        )

        # ── Planning ─────────────────────────────────────────────────────────
        opt_a, opt_steer, best_traj, predictions = planner.plan(world.ego, observations)

        # ── Physics Update ───────────────────────────────────────────────────
        world.step(opt_a, opt_steer)

        ego     = world.ego
        sim_t   = world.time
        real_t  = _time.time() - sim_start_real
        dist_km = world.travel_distance / 1000.0
        v_kmh   = ego.v * 3.6

        # Current speed zone
        zone      = world.get_current_speed_limit()
        lim_kmh   = zone.limit_mps * 3.6 if zone else None
        lim_label = zone.label if zone else '—'
        speeding  = zone and ego.v > zone.limit_mps + 0.5

        # ── MAIN VIEW ────────────────────────────────────────────────────────
        ax_main.clear()
        ax_main.set_facecolor(CLR_ROAD)
        ax_main.set_xlim(ego.x - 20, ego.x + 90)
        ax_main.set_ylim(-14, 14)
        ax_main.set_aspect('equal')
        ax_main.axis('off')

        # Road edges & center line
        ax_main.axhline( 5, color='white',    linewidth=2.5)
        ax_main.axhline(-5, color='white',    linewidth=2.5)
        ax_main.axhline( 0, color=CLR_LANE,   linestyle='--', linewidth=1.5, alpha=0.7)

        # Speed zone shading
        view_x0 = ego.x - 20
        view_x1 = ego.x + 90
        zone_colors = {'54 km/h': '#2ecc7122', '30 km/h': '#e74c3c22',
                       '50 km/h': '#f39c1222', '72 km/h': '#3498db22'}
        for z in world.speed_zones:
            zx0 = max(z.x_start, view_x0)
            zx1 = min(z.x_end,   view_x1)
            if zx0 < zx1:
                c = zone_colors.get(z.label, '#ffffff11')
                ax_main.fill_between([zx0, zx1], -5, 5, color=c, zorder=0)
                ax_main.text((zx0 + zx1) / 2, -6.5, z.label,
                             color='white', fontsize=7, ha='center', alpha=0.8)

        # Traffic lights
        for tl in world.traffic_lights:
            sc = signal_color(tl.state)
            ax_main.add_patch(plt.Rectangle((tl.x - 0.5, 5.5), 1.8, 5,  color='#555', zorder=3))
            ax_main.add_patch(plt.Circle((tl.x + 0.4, 9.2), 1.1, color=sc, zorder=4))
            ax_main.text(tl.x + 0.4, 7.0, tl.state, color='white', fontsize=6,
                         ha='center', va='center', zorder=5)
            ax_main.plot([tl.x, tl.x], [-5, 5], color='white', linewidth=2, alpha=0.8)

        # Stop signs
        for sign in world.stop_signs:
            ax_main.plot(sign.x, 6.5, marker='h', markersize=16, color=CLR_DANGER, zorder=4)
            ax_main.text(sign.x, 6.5, 'S', color='white', fontsize=7,
                         ha='center', va='center', fontweight='bold', zorder=5)
            ax_main.plot([sign.x, sign.x], [-5, 5], color=CLR_DANGER,
                         linewidth=2, alpha=0.7, linestyle='--')

        # Planned trajectory
        if best_traj:
            tx = [p[0] for p in best_traj]
            ty = [p[1] for p in best_traj]
            ax_main.plot(tx, ty, color=CLR_PLAN, linewidth=2.5, alpha=0.85, zorder=3)

        # Agent predictions
        for p_id, p_data in predictions.items():
            px = [p[0] for p in p_data['preds']]
            py = [p[1] for p in p_data['preds']]
            ax_main.plot(px, py, color=CLR_AGENT, linestyle=':', linewidth=1.5, alpha=0.5)

        # Ego vehicle
        ego_col = CLR_DANGER if speeding else CLR_EGO
        ax_main.add_patch(mpatches.FancyArrow(
            ego.x, ego.y,
            2.5 * np.cos(ego.yaw), 2.5 * np.sin(ego.yaw),
            width=1.8, head_width=2.2, head_length=1.5,
            color=ego_col, zorder=6
        ))
        ax_main.add_patch(plt.Rectangle(
            (ego.x - 2.5, ego.y - 0.9), 5, 1.8,
            color=ego_col, zorder=5, alpha=0.85
        ))

        # Dynamic agents
        for agent in world.agents:
            if 'Pedestrian' in type(agent).__name__:
                ax_main.add_patch(plt.Circle((agent.x, agent.y), agent.radius,
                                             color=CLR_PED, zorder=4))
            else:
                ax_main.add_patch(plt.Rectangle((agent.x - 2, agent.y - 1), 4, 2,
                                                color=CLR_AGENT, zorder=4))

        # Title
        spd_str = f'v: {v_kmh:.0f} km/h'
        lim_str = f'limit: {lim_label}' if lim_label else ''
        warn_str = '  ⚠ SPEEDING!' if speeding else ''
        ax_main.set_title(
            f'Autonomous Urban Driving  |  {spd_str}  {lim_str}{warn_str}  |  t: {sim_t:.1f}s',
            color=CLR_WARN if speeding else CLR_TEXT, fontsize=12, pad=6
        )

        # ── DASHBOARD ────────────────────────────────────────────────────────
        total_violations = (world.red_light_violations +
                            world.stop_sign_violations +
                            world.speeding_ticks // 10)
        law_status = 'COMPLIANT ✓' if total_violations == 0 else 'VIOLATION ✗'
        law_color  = CLR_OK if total_violations == 0 else CLR_DANGER

        # Format travel time (sim) and real elapsed time
        st_min, st_sec = divmod(int(sim_t), 60)
        rt_min, rt_sec = divmod(int(real_t), 60)

        col_vals = [
            f'SIM   {st_min:02d}:{st_sec:02d}\nREAL  {rt_min:02d}:{rt_sec:02d}',
            f'{world.travel_distance:>7.1f} m\n{dist_km:>7.3f} km',
            f'{v_kmh:>5.1f} km/h\nLIMIT {lim_label:>8s}',
            f'🔴 Red  {world.red_light_violations}\n'
            f'🛑 Stop {world.stop_sign_violations}\n'
            f'💨 Speed {world.speeding_ticks // 10}',
            f'{law_status}',
        ]
        col_colors = [CLR_TEXT, CLR_TEXT, CLR_DANGER if speeding else CLR_TEXT,
                      CLR_DANGER if total_violations > 0 else CLR_TEXT, law_color]

        for i, (val, col) in enumerate(zip(col_vals, col_colors)):
            dash_texts[i].set_text(val)
            dash_texts[i].set_color(col)

        # Violation log (last 4)
        recent = world.violations[-4:]
        for row in range(4):
            if row < len(recent):
                v = recent[-(row + 1)]
                vlog_texts[row].set_text(f"  [{v['time']:6.1f}s] {v['type']}")
            else:
                vlog_texts[row].set_text('')

    ani = animation.FuncAnimation(fig, animate, frames=600, interval=50, blit=False)
    plt.tight_layout(pad=0.3)
    plt.show()


if __name__ == '__main__':
    main()
