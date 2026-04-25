"""qlearning.py: TorchRL-registered Tabular Q-Learning.

Identical to the SB3-registered ``TabularQLearning`` but tagged as a TorchRL
algorithm (``is_torchrl = True``) for registry/dispatch purposes.  Q-learning
is tabular and doesn't use either SB3 or TorchRL infrastructure — the
distinction is purely organizational.

Usage::

    uv run python -m core.train --algorithm=torchrl-qlearning --env=resource \\
        --total-timesteps=50000 --no-eval --no-mlflow

    # Or directly:
    uv run python -m core.algorithms.torchrl.qlearning
"""

from core.algorithms.baselines.qlearning import TabularQLearning


class TorchRLTabularQLearning(TabularQLearning):
    """TorchRL-registered Tabular Q-Learning.

    Inherits all functionality from ``TabularQLearning``.  The only
    difference is ``is_torchrl = True`` so that ``train.py`` dispatches
    it through the tabular training path with the correct registry tag.
    """

    is_torchrl = True
    is_tabular = True


if __name__ == "__main__":
    print("=== TorchRL Tabular Q-Learning: Resource Gathering smoke test ===\n")
    model = TorchRLTabularQLearning(
        "resource",
        reward_component_names=["death_reward", "gold", "diamond"],
        component_weights=[0.2, 1.0, 0.5],
        seed=42,
    )
    metrics = model.learn(total_timesteps=5_000)
    print(f"\nSmoke test result: {metrics}")

    eval_metrics = model.evaluate(n_episodes=5)
    print(f"Eval result: {eval_metrics}")
    print("\nSmoke test passed!")
