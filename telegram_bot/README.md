# Kitchen Audio Telegram Bot

This folder contains the Telegram bot interface for the kitchen audio classification model. It accepts audio files or voice messages and predicts whether the sound is "Cooking" or "Not Cooking".

## Setup

1. **Get a Telegram Bot Token:**
   - Message [@BotFather](https://t.me/botfather) on Telegram.
   - Use the `/newbot` command to create a bot and get an HTTP API Token.

2. **Environment Configuration:**
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and insert your token: `TELEGRAM_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
   
3. **Install Dependencies:**
   Ensure you have installed the required packages (from the repository root):
   ```bash
   pip install -r requirements.txt
   pip install python-telegram-bot python-dotenv
   ```
   *(Note: You can also just run `make install` from the root).*

## Running the Bot

From the **root of the repository**, run:

```bash
# Export the token (or use python-dotenv inside your terminal)
source .env

# Run the bot
python -m telegram_bot.bot --ckpt results/checkpoints/best.pt
```

*(You can also use `make bot` from the repository root).*

## Usage

1. Open your bot in Telegram.
2. Send an audio file (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`) or record a Voice Message.
3. The bot will download the audio, process it through the dual-branch ensemble model, and return a prediction with confidence.
