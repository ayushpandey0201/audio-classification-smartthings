"""
Telegram bot for kitchen audio classification.
Accepts: audio file uploads (.wav / .mp3 / .ogg / .m4a) + voice messages.
Outputs: 🍳 Cooking  or  🔇 Not Cooking

Usage:
    python -m bot.bot --token YOUR_BOT_TOKEN --ckpt results/checkpoints/best.pt
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Add the parent directory to sys.path so we can import src
sys.path.append(str(Path(__file__).parent.parent))

from telegram_bot.inference import get_device, load_ensemble_model, predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Globals (set at startup) ──────────────────────────────────────────────────
MODEL = None
CONFIG = None
DEVICE = None
THRESHOLD = 0.5

# ── Handlers ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *Kitchen Audio Classifier*\n\n"
        "Send me an audio file or a voice message and I'll tell you whether "
        "it sounds like cooking.\n\n"
        "Supported formats: `.wav` `.mp3` `.ogg` `.m4a` `.flac`",
        parse_mode="Markdown",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Just send an audio file or record a voice message.\n"
        "I'll reply with 🍳 *Cooking* or 🔇 *Not Cooking*.",
        parse_mode="Markdown",
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle both Document audio uploads and Voice messages."""
    message = update.message

    if message.voice:
        tg_file = await message.voice.get_file()
        suffix = ".ogg"
    elif message.audio:
        tg_file = await message.audio.get_file()
        suffix = Path(message.audio.file_name or "audio.mp3").suffix or ".mp3"
    elif message.document:
        fname = message.document.file_name or ""
        ext = Path(fname).suffix.lower()
        if ext not in {".wav", ".mp3", ".ogg", ".m4a", ".flac"}:
            await message.reply_text(
                "⚠️ Please send a `.wav`, `.mp3`, `.ogg`, `.m4a`, or `.flac` file."
            )
            return
        tg_file = await message.document.get_file()
        suffix = ext
    else:
        await message.reply_text("⚠️ I can only process audio files or voice messages.")
        return

    await context.bot.send_chat_action(chat_id=message.chat_id, action="typing")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await tg_file.download_to_drive(tmp_path)
        logger.info(
            f"Downloaded audio to {tmp_path} ({os.path.getsize(tmp_path)} bytes)"
        )

        result = predict(tmp_path, MODEL, CONFIG, DEVICE, threshold=THRESHOLD)
        label = result["label"]

        if label == "Cooking":
            reply = f"🍳 *Cooking* (confidence: {result['prob']:.1%})"
        else:
            reply = f"🔇 *Not Cooking* (confidence: {1 - result['prob']:.1%})"

        await message.reply_text(reply, parse_mode="Markdown")
        logger.info(f"Classified: {label} (p={result['prob']:.3f})")

    except Exception as e:
        logger.exception("Inference error")
        await message.reply_text(f"❌ Error during classification: {e}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Please send an audio file or voice message. Type /help for instructions."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MODEL, CONFIG, DEVICE, THRESHOLD

    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Kitchen Audio Telegram Bot")
    parser.add_argument(
        "--token",
        help="Telegram Bot Token from @BotFather (or set TELEGRAM_TOKEN env var)",
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.pt")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Probability threshold (defaults to config evaluation.threshold)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise ValueError(
            "No token provided. Pass --token or set "
            "TELEGRAM_TOKEN environment variable."
        )

    DEVICE = get_device()
    logger.info(f"Using device: {DEVICE}")
    
    logger.info("Loading model...")
    MODEL, CONFIG = load_ensemble_model(args.ckpt, args.config, DEVICE)
    THRESHOLD = (
        args.threshold
        if args.threshold is not None
        else CONFIG["evaluation"]["threshold"]
    )
    
    logger.info(f"Model loaded. Threshold set to {THRESHOLD}")

    # Build application
    app = Application.builder().token(token).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.VOICE, handle_audio))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.Document.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, fallback))

    logger.info("Bot is polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
