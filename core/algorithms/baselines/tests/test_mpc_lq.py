"""LQMPCAgent unit tests — weight mapping, AR(1) fit, action mapping, warmup."""

from __future__ import annotations

import numpy as np
import pytest

from core.algorithms.baselines.mpc_lq import LQMPCAgent


def _agent_k4(**overrides):
    kwargs = dict(
        env_name="trading-k4",
        reward_component_names=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
        component_weights=None,
        horizon=10,
        history_window=200,
        warm_start_threshold=30,
    )
    kwargs.update(overrides)
    return LQMPCAgent(**kwargs)


def test_weight_mapping_k4_default():
    agent = _agent_k4()
    assert agent._w_pnl == 1.0
    assert agent._w_txn == 1.0
    assert agent._w_borrow == 1.0


def test_weight_mapping_k2():
    agent = LQMPCAgent(
        env_name="trading-k2",
        reward_component_names=["pnl", "costs"],
        component_weights=[2.0, 0.5],
        horizon=10,
    )
    assert agent._w_pnl == 2.0
    assert agent._w_txn == 0.5
    assert agent._w_borrow == 1.0


def test_weight_mapping_k8_float64_precision():
    agent = LQMPCAgent(
        env_name="trading-k8",
        reward_component_names=[
            "entry_gain",
            "entry_loss",
            "hold_gain",
            "hold_loss",
            "txn_cost",
            "borrow_cost",
            "spread_proxy",
            "residual",
        ],
        component_weights=[1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4],
    )
    # Strict tolerance: coefficients must be float64-precise (regression
    # against the float32-from-_init_component_weights bug).
    assert abs(agent._w_pnl - 2.5) < 1e-12
    assert abs(agent._w_txn - 0.2) < 1e-12
    assert abs(agent._w_borrow - 0.3) < 1e-12


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError, match="component weights"):
        LQMPCAgent(
            env_name="trading-k4",
            reward_component_names=["pnl_gain", "pnl_loss", "txn_cost", "borrow_cost"],
            component_weights=[1.0, 2.0],
        )


def test_warmup_returns_flat():
    agent = _agent_k4(warm_start_threshold=30)
    info = {"data_close": 100.0, "position": 0}
    for _ in range(5):
        a = agent.act(info)
        assert a == 1


def test_act_with_empty_info_uses_cached_fallback():
    agent = _agent_k4(warm_start_threshold=30)
    a = agent.act({})
    assert a in (0, 1, 2)


def test_ar1_recovers_true_ou_parameters():
    rng = np.random.default_rng(42)
    n = 500
    mu = 100.0
    theta = 0.05  # → phi = 0.95
    sigma = 3.0
    x = np.zeros(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()

    agent = _agent_k4(history_window=500)
    for v in x:
        agent._price_buffer.append(float(v))
    agent._fit_ar1()
    assert 0.85 <= agent._ar_phi <= 0.99
    assert 90.0 <= agent._ar_mu <= 110.0


def test_continuous_to_discrete_thresholds():
    fn = LQMPCAgent._continuous_to_discrete
    assert fn(-1.0) == 0
    assert fn(-0.34) == 0
    assert fn(-0.33) == 1
    assert fn(0.0) == 1
    assert fn(0.33) == 1
    assert fn(0.34) == 2
    assert fn(1.0) == 2


def test_predict_returns_action_and_none():
    agent = _agent_k4()
    obs = np.zeros(5)
    a, state = agent.predict(obs)
    assert isinstance(a, int)
    assert state is None


def test_class_flags_match_train_dispatch():
    """train.py routes via getattr(ModelClass, "is_tabular", ...) etc."""
    assert LQMPCAgent.is_tabular is True
    assert LQMPCAgent.is_torchrl is False
    assert LQMPCAgent.is_self_normalizing is False


def test_parser_mpc_horizon_flag():
    """--mpc-horizon flag parses to args.mpc_horizon (None when absent)."""
    from core.parser import parse_args

    args = parse_args(["--algorithm=mpc-lq", "--env=trading-k4", "--mpc-horizon=15"])
    assert args.mpc_horizon == 15

    args_default = parse_args(["--algorithm=mpc-lq", "--env=trading-k4"])
    assert args_default.mpc_horizon is None


def test_horizon_kwarg_in_signature():
    """train.py introspects ModelClass.__init__ for 'horizon'; must be present."""
    import inspect

    sig = inspect.signature(LQMPCAgent.__init__)
    assert "horizon" in sig.parameters
