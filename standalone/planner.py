import numpy as np
from standalone.kinematics import BicycleModel

class MultiObjectivePlanner:
    def __init__(self, target_speed=15.0):
        self.target_speed = target_speed

    def generate_trajectories(self, ego_state, dt=0.1, horizon=25):
        """Generate candidate trajectory rollouts."""
        trajectories = []
        for steer in np.linspace(-np.radians(10), np.radians(10), 5):
            for acc in np.linspace(-4.0, 2.0, 7):
                sim_ego = BicycleModel(ego_state.x, ego_state.y, ego_state.yaw, ego_state.v)
                traj = []
                for _ in range(horizon):
                    sim_ego.update(acc, steer, dt)
                    traj.append((sim_ego.x, sim_ego.y, sim_ego.v, sim_ego.yaw, acc, steer))
                trajectories.append(traj)
        return trajectories

    def predict_agents(self, obs_agents, dt=0.1, horizon=25):
        """Constant-velocity prediction for dynamic agents."""
        predictions = {}
        for agent in obs_agents:
            preds = []
            x, y = agent['x'], agent['y']
            for _ in range(horizon):
                x += agent['vx'] * dt
                y += agent['vy'] * dt
                preds.append((x, y))
            predictions[agent['id']] = {'preds': preds, 'radius': agent['radius']}
        return predictions

    def evaluate_cost(self, traj, observations, predictions, ego_v, dt=0.1):
        cost = 0.0

        # Current speed limit (from observations, falls back to self.target_speed)
        speed_limit = observations.get('speed_limit') or self.target_speed
        effective_target = min(self.target_speed, speed_limit)

        # ── 1. Comfort ──────────────────────────────────────────────────────────
        accels = [p[4] for p in traj]
        steers = [p[5] for p in traj]
        cost += np.sum(np.abs(np.diff(accels))) * 5.0
        cost += np.sum(np.abs(np.diff(steers))) * 20.0

        for i, point in enumerate(traj):
            x, y, v, yaw, a, s = point

            # ── 2. Efficiency ─────────────────────────────────────────────────
            cost += abs(effective_target - v) * 2.0

            # ── 3. Speed limit hard cap ───────────────────────────────────────
            if v > speed_limit + 0.5:
                # Massive penalty for every m/s over the limit
                cost += 800.0 * (v - speed_limit)

            # ── 4. Lane keeping ───────────────────────────────────────────────
            cost += abs(y) * 10.0
            cost += abs(yaw) * 5.0
            cost += abs(s)  * 10.0

            # Anti-stuck penalty
            if v < 1.0:
                cost += 300.0 * (1.0 - v)

            # ── 5. Safety ─────────────────────────────────────────────────────
            for p_id, p_data in predictions.items():
                agent_pred_x, agent_pred_y = p_data['preds'][i]
                dist = np.hypot(x - agent_pred_x, y - agent_pred_y)
                if dist < p_data['radius'] + 2.0:
                    return float('inf')
                elif dist < 6.0:
                    cost += 2000.0 / dist

            # ── 6. Traffic-Law Adherence ──────────────────────────────────────
            for tl in observations['lights']:
                # Running red / yellow = infinite cost (hard constraint)
                if x >= tl['x'] > traj[0][0] and tl['state'] in ['RED', 'YELLOW']:
                    return float('inf')
                # Smooth deceleration zone: start braking 20 m before red light
                if tl['state'] == 'RED' and tl['x'] - 20.0 < x < tl['x']:
                    cost += 500.0 * v   # strongly penalise high speed near red light
                # Yellow light: decelerate if close enough to stop safely
                if tl['state'] == 'YELLOW' and tl['x'] - 15.0 < x < tl['x']:
                    cost += 300.0 * v

            for sign in observations['signs']:
                dist_to_sign = sign['x'] - x
                # Approach zone: slow down progressively
                if 0 < dist_to_sign < 20.0:
                    cost += 40.0 * v * (1.0 - dist_to_sign / 20.0)
                # At the sign: must be near stop
                elif -2.0 < dist_to_sign <= 0:
                    if v > 0.5:
                        cost += 6000.0

        return cost

    def plan(self, ego_state, observations):
        trajectories = self.generate_trajectories(ego_state)
        predictions  = self.predict_agents(observations['agents'])

        best_traj = None
        best_cost = float('inf')

        for traj in trajectories:
            c = self.evaluate_cost(traj, observations, predictions, ego_state.v)
            if c < best_cost:
                best_cost = c
                best_traj = traj

        if best_traj is None:
            return -4.0, 0.0, None, predictions

        opt_a     = best_traj[0][4]
        opt_steer = best_traj[0][5]
        return opt_a, opt_steer, best_traj, predictions
