"""
main.py — Entry point for the Multi-Agent RL Civilization Simulation.

Usage:
    python main.py                  # Run with Pygame visualization
    python main.py --headless       # Run without visualization
    python main.py --steps 2000     # Override max steps
    python main.py --population 120 # Override initial population
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from config import WorldConfig
from world import World
from policies import PolicyPool
from renderer import Renderer
from analytics import plot_metrics, plot_policy_fitness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent RL Civilization Simulation")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--population", type=int, default=None, help="Override initial population")
    parser.add_argument("--width", type=int, default=None, help="Grid width")
    parser.add_argument("--height", type=int, default=None, help="Grid height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pool-size", type=int, default=10, help="Policy pool size")
    parser.add_argument("--policy", type=str, default="neural",
                        choices=["neural", "tabular", "rule_based"],
                        help="Policy representation type")
    parser.add_argument("--log-interval", type=int, default=100, help="Console log every N steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build config ──────────────────────────────────────────────────
    config = WorldConfig(terrain_seed=args.seed)
    if args.steps is not None:
        config.max_steps = args.steps
    if args.population is not None:
        config.initial_population = args.population
    if args.width is not None:
        config.width = args.width
    if args.height is not None:
        config.height = args.height

    if args.headless:
        os.environ["RENDER_HEADLESS"] = "1"

    # ── Initialize ────────────────────────────────────────────────────
    world = World(config)
    pool = PolicyPool(
        size=args.pool_size,
        kind=args.policy,
        observation_radius=config.observation_radius,
    )

    # Assign random policies to all agents after reset
    obs = world.reset()
    for a in world.agents:
        a.policy_id = pool.assign_random()

    # ── Renderer ──────────────────────────────────────────────────────
    renderer = Renderer(config)
    use_renderer = renderer.init()

    # ── Simulation loop ───────────────────────────────────────────────
    policy_stats_history = []
    t_start = time.time()

    print(f"[Simulation] Starting — {config.width}x{config.height} grid, "
          f"{config.initial_population} agents, {config.max_steps} max steps")
    print(f"[Simulation] Policy: {args.policy} | Pool size: {args.pool_size}")
    print(f"[Simulation] Renderer: {'ON' if use_renderer else 'HEADLESS'}")

    for step_i in range(config.max_steps):
        # Get actions from policy pool
        alive = world.alive_agents
        if not alive:
            print(f"[Simulation] Extinction at step {step_i}!")
            break

        prev_obs = obs
        actions = pool.select_actions(alive, obs)

        # Step the world
        result = world.step(actions)
        obs = result["observations"]
        rewards = result["rewards"]
        info = result["info"]

        # Store transitions and learn
        pool.store_transitions(
            alive, prev_obs, actions, rewards, obs, result["done"]
        )
        pool.learn_all()

        # Update policy fitness
        pool.update_fitness(rewards, world.agents)
        policy_stats_history.append(pool.get_stats())

        # Render
        if use_renderer:
            for _ in range(config.steps_per_frame - 1):
                # Fast-forward N-1 steps without rendering
                if not world.alive_agents:
                    break
                a2 = pool.select_actions(world.alive_agents, obs)
                r2 = world.step(a2)
                obs = r2["observations"]
                pool.update_fitness(r2["rewards"], world.agents)

            keep_running = renderer.render(
                world.terrain, world.food, world.alive_agents, info
            )
            if not keep_running:
                print("[Simulation] Window closed by user.")
                break

        # Console log
        if step_i % args.log_interval == 0:
            elapsed = time.time() - t_start
            sps = (step_i + 1) / max(elapsed, 0.001)
            print(
                f"  Step {step_i:>5} | Alive: {info['alive']:>4} | "
                f"F:{info['class_counts']['FARMER']:>3} W:{info['class_counts']['WARRIOR']:>3} "
                f"T:{info['class_counts']['TRADER']:>3} L:{info['class_counts']['LEADER']:>3} | "
                f"Food: {info['avg_food']:.1f} HP: {info['avg_health']:.1f} | "
                f"{sps:.0f} steps/s"
            )

        if result["done"]:
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    renderer.close()
    elapsed = time.time() - t_start
    final_step = world.current_step
    print(f"\n[Simulation] Finished — {final_step} steps in {elapsed:.1f}s "
          f"({final_step / max(elapsed, 0.001):.0f} steps/s)")

    # ── Analytics ─────────────────────────────────────────────────────
    print("[Analytics] Generating dashboards...")
    plot_metrics(world.metrics, save_path="analytics.png")
    plot_policy_fitness(policy_stats_history, save_path="policy_fitness.png")
    print("[Done]")


if __name__ == "__main__":
    main()
