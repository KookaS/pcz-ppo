"""StaticWeightAgent unit tests — class flags, z-score thresholds, warmup."""

from __future__ import annotations

from core.algorithms.baselines.static_weight_agent import StaticWeightAgent


def _agent(**overrides):
    kwargs = dict(
        env_name="trading-k4",
        reward_component_names=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
        threshold=0.5,
        history_window=200,
        warm_start_threshold=30,
        seed=42,
    )
    kwargs.update(overrides)
    return StaticWeightAgent(**kwargs)


def test_class_flags_match_train_dispatch():
    assert StaticWeightAgent.is_tabular is True
    assert StaticWeightAgent.is_torchrl is False
    assert StaticWeightAgent.is_self_normalizing is False


def test_warmup_returns_flat():
    agent = _agent(warm_start_threshold=30)
    info = {"data_close": 100.0}
    for _ in range(5):
        a = agent.act(info)
        assert a == 1


def test_zscore_high_returns_short():
    """z > +threshold → action 0 (short)."""
    agent = _agent(threshold=0.5, warm_start_threshold=30)
    for _ in range(40):
        agent._price_buffer.append(100.0)
    # Inject one high-z observation
    a = agent.act({"data_close": 110.0})
    assert a == 0


def test_zscore_low_returns_long():
    """z < -threshold → action 2 (long)."""
    agent = _agent(threshold=0.5, warm_start_threshold=30)
    for _ in range(40):
        agent._price_buffer.append(100.0)
    a = agent.act({"data_close": 90.0})
    assert a == 2


def test_zscore_within_band_returns_flat():
    """|z| <= threshold → action 1 (flat). Constant buffer → std≈0 → z=0."""
    agent = _agent(threshold=0.5, warm_start_threshold=30)
    for _ in range(40):
        agent._price_buffer.append(100.0)
    a = agent.act({"data_close": 100.0})
    assert a == 1


def test_constant_buffer_does_not_divide_by_zero():
    """std≈0 in buffer → safe early-return as flat."""
    agent = _agent(warm_start_threshold=30)
    for _ in range(40):
        agent._price_buffer.append(100.0)
    a = agent.act({"data_close": 100.0})
    assert a == 1


def test_predict_returns_action_and_none():
    agent = _agent()
    a, state = agent.predict(obs=None)
    assert isinstance(a, int)
    assert state is None


def test_threshold_persisted_in_save_load(tmp_path):
    agent = _agent(threshold=0.7)
    agent.save(str(tmp_path))
    fresh = _agent(threshold=0.5)
    fresh.load(str(tmp_path))
    assert fresh.threshold == 0.7


def test_registry_entry():
    from core import ALGORITHM_REGISTRY

    assert "static-weight" in ALGORITHM_REGISTRY
    assert ALGORITHM_REGISTRY["static-weight"] is StaticWeightAgent
