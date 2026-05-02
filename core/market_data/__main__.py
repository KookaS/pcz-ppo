"""Allow ``python -m core.market_data ...`` to dispatch to ``cli.main``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
