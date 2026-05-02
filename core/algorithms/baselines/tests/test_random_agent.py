"""RandomAgent unit tests — class flags, action sampling, interface contract."""

from __future__ import annotations

from core.algorithms.baselines.random_agent import RandomAgent


def _agent(**overrides):
    kwargs = dict(
        env_name="trading-k4",
        reward_component_names=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
        seed=42,
    )
    kwargs.update(overrides)
    return RandomAgent(**kwargs)


def test_class_flags_match_train_dispatch():
    """train.py routes via getattr(ModelClass, "is_tabular", ...) etc."""
    assert RandomAgent.is_tabular is True
    assert RandomAgent.is_torchrl is False
    assert RandomAgent.is_self_normalizing is False


def test_predict_returns_action_and_none():
    agent = _agent()
    a, state = agent.predict(obs=None)
    assert isinstance(a, int)
    assert 0 <= a <= 2
    assert state is None


def test_act_uses_action_space_sample():
    """act() forwards to action_space.sample()."""

    class _FakeSpace:
        def __init__(self):
            self.calls = 0

        def sample(self):
            self.calls += 1
            return 2

    space = _FakeSpace()
    agent = _agent()
    a = agent.act(space)
    assert a == 2
    assert space.calls == 1


def test_seed_determinism_via_predict():
    """Two agents with the same seed produce identical predict() sequences."""
    a1 = _agent(seed=7)
    a2 = _agent(seed=7)
    s1 = [a1.predict(obs=None)[0] for _ in range(50)]
    s2 = [a2.predict(obs=None)[0] for _ in range(50)]
    assert s1 == s2


def test_registry_entry():
    """Registered as 'random-action'."""
    from core import ALGORITHM_REGISTRY

    assert "random-action" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["random-action"] is RandomAgent
