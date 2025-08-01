#!/usr/bin/env python3
"""
Простий Telegram-бот із командами /start, /help та /echo.
Побудовано на python-telegram-bot 13.15 (Long Polling).
"""

import logging
from telegram import Update, ParseMode
from telegram.ext import (
    Updater,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    Filters,
)

# ——— 1. ВСТАВТЕ СВІЙ ТОКЕН НИЖЧЕ ———
TOKEN: str = "8358645388:AAFmhfpSAIyB-ha8vCh-MTu2YeYKVT8fmms"
# ————————————————————————————————

# Увімкнути детальний логгер (видно в консолі)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# /start
def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr"Привіт, *{user.first_name}*!\n"
        "Я простий демонстраційний бот\. "
        "Надішліть /help, щоб побачити доступні команди\.",
    )


# /help
def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        "Доступні команди:\n"
        "/start – вітання\n"
        "/help – довідка\n"
        "/echo <текст> – повторити ваш текст"
    )


# /echo <text…>
def echo(update: Update, context: CallbackContext) -> None:
    if context.args:
        update.message.reply_text(" ".join(context.args))
    else:
        update.message.reply_text("Ви не передали текст після /echo.")


# Ловимо всі необроблені помилки, щоб бот не впав
def error_handler(update: object, context: CallbackContext) -> None:
    logger.error("Помилка у виклику: %s", context.error)


def main() -> None:
    """Точка входу."""
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    # Реєструємо командні обробники
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("echo", echo, pass_args=True))

    # Обробник усіх інших текстових повідомлень (не обов’язково)
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # Логування помилок
    dp.add_error_handler(error_handler)

    # Запускаємо Long Polling
    updater.start_polling()
    logger.info("Бот запущено. Натисніть Ctrl+C для зупинки.")
    updater.idle()


if __name__ == "__main__":
    main()
