"""Adversarial fixture: a figure script with mixed UNITS but missing
``MIXED_UNITS_ACKNOWLEDGED = True``.

Verifies that ``lint_figure_units.py`` flags the missing acknowledgment.
"""

INPUTS = []
UNITS = "rollout-per-step,eval-episodic"  # mixed but no MIXED_UNITS_ACKNOWLEDGED
OUTPUTS = ["fig_bad_mixed_units.pdf"]


def main():
    pass
