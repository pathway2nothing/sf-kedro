try:
    from .telegram import send_plots_to_telegram
except ImportError:
    # telebot not installed
    def send_plots_to_telegram(*args, **kwargs):
        raise ImportError(
            "pyTelegramBotAPI is not installed. "
            "Install it with: pip install pyTelegramBotAPI"
        )


__all__ = [
    "send_plots_to_telegram",
]
