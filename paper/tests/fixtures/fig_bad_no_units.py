"""Adversarial fixture: a figure script with NO ``UNITS`` declaration.

Verifies that ``lint_figure_units.py`` flags the missing declaration.
"""

INPUTS = []
# Intentionally NO UNITS declaration here.
OUTPUTS = ["fig_bad_no_units.pdf"]


def main():
    pass
