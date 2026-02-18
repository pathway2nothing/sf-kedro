try:
    from .telegram import send_message_to_telegram, send_plots_to_telegram
except ImportError:
    # telebot not installed
    def send_plots_to_telegram(*args, **kwargs):
        raise ImportError("pyTelegramBotAPI is not installed. Install it with: pip install pyTelegramBotAPI")

    def send_message_to_telegram(*args, **kwargs):
        raise ImportError("pyTelegramBotAPI is not installed. Install it with: pip install pyTelegramBotAPI")


from .detection import run_detection, run_detection_with_detector

__all__ = [
    "run_detection",
    "run_detection_with_detector",
    "send_message_to_telegram",
    "send_plots_to_telegram",
]
