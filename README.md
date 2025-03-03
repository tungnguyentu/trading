# Crypto Trading Simulator

A real-time cryptocurrency trading simulator that uses technical indicators and machine learning to simulate trading strategies on Binance.

## Features

- Real-time price data from Binance API
- Technical analysis with moving averages and ATR
- Machine learning signal enhancement
- Telegram notifications for trade signals and position updates
- Customizable trading parameters (leverage, investment amount, profit targets)
- Detailed performance reports and visualizations

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd snake
```
2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the project root with your Binance API keys and Telegram credentials:
```bash
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Usage
### Running the Real-time Simulator
```bash
python realtime_simulation.py --symbol BTCUSDT --investment 50 --target 15 --hours 24 --interval 15 --leverage 15
 ```

### Command Line Arguments
- --symbol : Trading pair to simulate (default: BTCUSDT)
- --investment : Initial investment amount in USD (default: 50.0)
- --target : Daily profit target in USD (default: 15.0)
- --hours : Duration of simulation in hours (default: 24)
- --interval : Update interval in minutes (default: 15)
- --leverage : Margin trading leverage (15-20x) (default: 15)
## Example
```bash
# Run a 48-hour simulation on ETHUSDT with $100 investment
python realtime_simulation.py --symbol ETHUSDT --investment 100 --target 20 --hours 48 --interval 30 --leverage 20
```

## Telegram Notifications
The simulator sends real-time notifications to your Telegram account:

- When a simulation starts
- When a position is opened
- Regular updates on open positions
- When a position is closed (take profit/stop loss)
- When the simulation ends
Make sure your Telegram bot token and chat ID are correctly set in the .env file to receive these notifications.

## Results
The simulation results will be saved in the realtime_results directory, including:

- Trade history
- Equity curve
- Performance metrics
- Daily profit/loss
## Notes
- This is a simulation tool and does not execute real trades
- Use at your own risk and always do your own research
- Past performance is not indicative of future results