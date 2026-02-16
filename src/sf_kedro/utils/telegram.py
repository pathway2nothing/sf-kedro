# sf_kedro/utils/telegram.py

import os
from typing import List, Dict, Optional, TYPE_CHECKING
from pathlib import Path
import tempfile
from io import BytesIO
from loguru import logger
import plotly.graph_objects as go

try:
    import telebot
    from telebot.types import InputMediaPhoto
    TELEBOT_AVAILABLE = True
except ImportError:
    TELEBOT_AVAILABLE = False
    telebot = None
    InputMediaPhoto = None


class TelegramNotifier:
    """Send plots and messages to Telegram channel using pyTelegramBotAPI."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token (from @BotFather)
            chat_id: Channel ID (e.g., "@my_channel" or "-100123456789")
        """
        if not TELEBOT_AVAILABLE:
            raise ImportError(
                "pyTelegramBotAPI is not installed. "
                "Install it with: pip install pyTelegramBotAPI"
            )

        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "Telegram bot_token and chat_id must be provided either "
                "as arguments or via TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars"
            )

        self.bot = telebot.TeleBot(self.bot_token)
        logger.info(f"Telegram bot initialized for chat: {self.chat_id}")
    
    def send_message(self, text: str) -> None:
        """Send text message to channel."""
        try:
            self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="HTML"
            )
            logger.info(f"Sent message to Telegram: {text[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def send_plot(
        self,
        fig: go.Figure,
        caption: Optional[str] = None,
        width: int = 1400,
        height: int = 900,
    ) -> None:
        """Send single plot to channel."""
        try:
            # Save to BytesIO instead of temp file
            img_bytes = BytesIO()
            fig.write_image(img_bytes, format='png', width=width, height=height, scale=2)
            img_bytes.seek(0)
            
            self.bot.send_photo(
                chat_id=self.chat_id,
                photo=img_bytes,
                caption=caption,
                parse_mode="HTML"
            )
            logger.info(f"Sent plot to Telegram: {caption or 'no caption'}")
        except Exception as e:
            logger.error(f"Failed to send plot: {e}")
    
    def send_plots_group(
        self,
        figures: List[go.Figure],
        metric_name: str,
        width: int = 1400,
        height: int = 900,
        max_per_group: int = 10,
    ) -> None:
        """
        Send multiple plots as media group (album).
        
        Telegram limits media groups to 10 items, so we split if needed.
        
        Args:
            figures: List of plotly figures
            metric_name: Name of metric for caption
            width: Image width
            height: Image height
            max_per_group: Max images per album (Telegram limit is 10)
        """
        if not figures:
            logger.warning(f"No figures to send for {metric_name}")
            return
        
        # Split into chunks if more than max_per_group
        total_chunks = (len(figures) + max_per_group - 1) // max_per_group
        
        for chunk_idx, start_idx in enumerate(range(0, len(figures), max_per_group)):
            chunk = figures[start_idx:start_idx + max_per_group]
            
            media_group = []
            
            try:
                # Create media group
                for i, fig in enumerate(chunk):
                    img_bytes = BytesIO()
                    fig.write_image(
                        img_bytes,
                        format='png',
                        width=width,
                        height=height,
                        scale=2
                    )
                    img_bytes.seek(0)
                    
                    # First image gets caption
                    caption = None
                    if i == 0:
                        if total_chunks > 1:
                            caption = (
                                f"<b>{metric_name}</b>\n"
                                f"Part {chunk_idx + 1}/{total_chunks}"
                            )
                        else:
                            caption = f"<b>{metric_name}</b>"
                    
                    media_group.append(
                        InputMediaPhoto(
                            media=img_bytes,
                            caption=caption,
                            parse_mode="HTML"
                        )
                    )
                
                # Send media group
                self.bot.send_media_group(
                    chat_id=self.chat_id,
                    media=media_group
                )
                
                logger.info(
                    f"Sent {len(chunk)} plots for {metric_name} "
                    f"(part {chunk_idx + 1}/{total_chunks})"
                )
                
            except Exception as e:
                logger.error(f"Failed to send media group for {metric_name}: {e}")
                # Fallback: send individually
                logger.info("Attempting to send plots individually...")
                for i, fig in enumerate(chunk):
                    try:
                        caption = f"<b>{metric_name}</b> - Plot {start_idx + i + 1}/{len(figures)}"
                        self.send_plot(fig, caption=caption, width=width, height=height)
                    except Exception as inner_e:
                        logger.error(f"Failed to send individual plot {i}: {inner_e}")
    
    def send_all_metrics_plots(
        self,
        plots: Dict[str, List[go.Figure] | go.Figure],
        header_message: Optional[str] = None,
        width: int = 1400,
        height: int = 900,
    ) -> None:
        """
        Send all metric plots to Telegram, grouped by metric.
        
        Args:
            plots: Dict mapping metric names to figures
            header_message: Optional header message to send first
            width: Image width
            height: Image height
        """
        if header_message:
            self.send_message(header_message)
        
        total_metrics = len(plots)
        for idx, (metric_name, figures) in enumerate(plots.items(), 1):
            if figures is None:
                logger.warning(f"Skipping {metric_name}: no figures")
                continue
            
            # Normalize to list
            if not isinstance(figures, list):
                figures = [figures]
            
            if len(figures) == 0:
                logger.warning(f"Skipping {metric_name}: empty list")
                continue
            
            logger.info(f"Sending metric {idx}/{total_metrics}: {metric_name} ({len(figures)} plots)")
            
            # Send as group if multiple plots, single if just one
            if len(figures) == 1:
                self.send_plot(
                    fig=figures[0],
                    caption=f"<b>{metric_name}</b>",
                    width=width,
                    height=height
                )
            else:
                self.send_plots_group(
                    figures=figures,
                    metric_name=metric_name,
                    width=width,
                    height=height
                )
        
        logger.info(f"Completed sending {total_metrics} metrics to Telegram")


def send_plots_to_telegram(
    plots: Dict[str, List[go.Figure] | go.Figure],
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    header_message: Optional[str] = None,
    image_width: int = 1400,
    image_height: int = 900,
) -> None:
    """
    Synchronous function for sending plots to Telegram.
    
    Args:
        plots: Dictionary with {metric_name: [figures]}
        bot_token: Telegram bot token (or set TELEGRAM_BOT_TOKEN env var)
        chat_id: Channel ID (or set TELEGRAM_CHAT_ID env var)
        header_message: Optional header message
        image_width: Width of images in pixels
        image_height: Height of images in pixels
    """
    try:
        notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
        notifier.send_all_metrics_plots(
            plots=plots,
            header_message=header_message,
            width=image_width,
            height=image_height
        )
        logger.info("Successfully sent all plots to Telegram")
    except Exception as e:
        logger.error(f"Failed to send plots to Telegram: {e}")
        raise