from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Launch the Streamlit research dashboard."""
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install the dashboard extras with `pip install -e .[dashboard]`."
        ) from exc

    dashboard_path = Path(__file__).with_name("dashboard.py")
    sys.argv = ["streamlit", "run", str(dashboard_path)]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
