# Installation Guide

This guide will help you set up the trading bot with all its dependencies.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Basic Installation

1. Clone or download the repository:
   ```
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Binance API keys:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token (optional)
   TELEGRAM_CHAT_ID=your_telegram_chat_id (optional)
   ```

4. Run the bot with default settings:
   ```
   python run_realtime_trading.py
   ```

## Technical Indicators Installation

The bot supports multiple technical indicator libraries with a fallback system:

### Option 1: Use Alternative Libraries (Easiest)

The bot will automatically use pandas_ta or ta if TA-Lib is not available. These are pure Python libraries that are installed automatically with the requirements.txt file.

### Option 2: Install TA-Lib (Recommended for Performance)

TA-Lib provides better performance but requires a C/C++ library to be installed:

#### For macOS (using Homebrew):
```
brew install ta-lib
pip install TA-Lib
```

#### For Ubuntu/Debian:
```
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

#### For Windows:
1. Download the pre-built binary from: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip
2. Extract to `C:\ta-lib`
3. Install the Python wrapper:
   ```
   pip install --find-links=https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib TA-Lib
   ```

## Docker Installation (Alternative)

If you're having trouble with dependencies, you can use Docker:

1. Build the Docker image:
   ```
   docker build -t crypto-trading-bot .
   ```

2. Run the container:
   ```
   docker run -it --env-file .env crypto-trading-bot python run_realtime_trading.py
   ```

## Troubleshooting

### TA-Lib Installation Issues

If you're having trouble installing TA-Lib:

1. The bot will automatically fall back to alternative libraries
2. You can explicitly install the alternatives:
   ```
   pip install pandas-ta ta
   ```

### Binance API Connection Issues

1. Make sure your API keys are correct and have the necessary permissions
2. Check your internet connection
3. Verify that your IP is not restricted in Binance API settings

### Python Version Issues

The bot is tested with Python 3.7-3.9. If you're using a different version:

1. Consider using a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Next Steps

After installation, check out the following resources:

- `README.md` for general usage information
- `READ.md` for detailed settings documentation
- `migrate_to_modular.py` for information about the new modular structure 