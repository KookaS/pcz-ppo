"""Metrics export, diagnostics, and visualization tools for PCZ-PPO experiments.

Modules:
    export_results        — Lightweight CSV index (1 row/run, final metrics + hyperparams)
    export_metrics        — Full time-series export to parquet (1 file/run)
    diagnostics           — Automated post-training analysis (spikes, convergence, trust)
    component_correlation — Pairwise Pearson correlation heatmap for reward components
"""
