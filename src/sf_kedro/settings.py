# src/sf_kedro/settings.py

from pathlib import Path

# Load .env if exists
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        pass

from sf_kedro.hooks.dagshub_hooks import DagsHubHook

HOOKS = (DagsHubHook(),)

DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)