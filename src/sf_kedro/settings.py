from pathlib import Path

env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        pass

from sf_kedro.hooks.dagshub_hooks import DagsHubHook  # noqa: E402

HOOKS = (DagsHubHook(),)

DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)
