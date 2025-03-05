# Crypto Trading Bot

This is a cryptocurrency trading bot that can run in both simulation mode and real trading mode using the Binance API.

## Features

- Real-time trading on Binance
- Simulation mode for testing strategies
- Test mode with fake balance for safe testing of real trading functionality
- Technical analysis indicators (SMA, ATR)
- Machine learning signal integration
- Risk management (stop loss, take profit)
- Market orders for immediate execution at current price
- Telegram notifications
- Detailed trade history and reporting

## Requirements

- Python 3.7+
- Binance account with API keys
- Telegram bot (optional, for notifications)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Usage

### Simulation Mode

To run in simulation mode:

```
python run_simulation.py --symbol BTCUSDT --investment 50 --leverage 15 --hours 24 --interval 15
```

### Test Mode

To run in test mode with fake balance (no real trades executed):

```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 15 --hours 24 --interval 15 --test
```

### Real Trading Mode

To run in real trading mode:

```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 15 --hours 24 --interval 15
```

### Full Margin Mode (EXTREME RISK)

To run in full margin mode (where your entire investment is used as margin):

```
python run_realtime_trading.py --symbol SOLUSDT --investment 50 --leverage 20 --hours 24 --interval 15 --full-margin
```

This mode will use your entire $50 as margin, controlling a position worth $1,000 (with 20x leverage). The margin shown in Binance will be $50.

### Command Line Arguments

- `--symbol`: Trading pair (default: BTCUSDT)
- `--investment`: Initial investment amount in USD (default: 50.0)
- `--target`: Daily profit target in USD (default: 15.0) - When reached, the bot will close the current position
- `--hours`: Duration in hours (default: 24)
- `--interval`: Update interval in minutes (default: 15)
- `--leverage`: Margin trading leverage (default: 15, range: 15-20)
- `--test`: Run in test mode with fake balance (no real trades)
- `--full-investment`: Use full investment amount for each trade (higher risk)
- `--full-margin`: Use full investment amount as margin (EXTREME RISK)
- `--compound`: Enable compound interest (use profits to increase position sizes)

## Risk Warning

**IMPORTANT**: This bot trades with real money when in real trading mode. Use at your own risk. The authors are not responsible for any financial losses incurred from using this software.

**EXTRA CAUTION**: The `--full-investment` flag makes the bot use your entire investment amount for each trade. This significantly increases risk and could lead to losing your entire investment quickly. Only use this option if you fully understand the risks involved.

**EXTREME RISK WARNING**: The `--full-margin` flag makes the bot use your entire investment amount as margin. This means your entire investment will be at risk of liquidation. With 20x leverage, a mere 5% price movement against your position will result in complete loss of your investment. Only use this option if you fully understand and accept the extreme risks involved.

## Compound Interest

The bot includes a compound interest feature that can potentially increase your returns over time:

1. When enabled, the bot will use your current account balance (including profits) to calculate position sizes
2. As your balance grows, your position sizes will increase proportionally
3. This can lead to exponential growth of your account if you're consistently profitable
4. However, it also increases risk as losses will be larger with bigger position sizes

How compound interest works with different trading modes:

- **Standard Mode**: Uses your current account balance to calculate risk amount and position sizes
- **Full Investment Mode**: Uses your current account balance for position sizing
- **Full Margin Mode**: Always uses your initial investment amount as margin, regardless of compound interest setting

To enable compound interest, use the `--compound` flag:
```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 15 --compound --test
```

This feature works with all trading modes and can be combined with other flags.

## Safety Features

- Test mode with fake balance for safe testing
- Confirmation prompt before starting real trading
- Daily profit target to secure profits (closes position when target is reached)
- Daily loss limit (10% of initial investment by default)
- Trading disabled automatically when daily loss limit is reached
- Position size limited to a percentage of account balance
- Risk per trade limited to a percentage of account balance
- Tighter stop loss (1.5x ATR, capped at 3%) and take profit (2x ATR, capped at 6%) levels
- Automatic handling of asset precision requirements
- Market orders for immediate execution at best available price
- Real stop loss and take profit orders placed directly on Binance (executes immediately when price is reached)

## Daily Profit Target

The bot includes a daily profit target feature that helps secure profits:

1. When your daily profit reaches the target amount (default: $15), the bot will close any open position
2. This helps lock in profits and prevents giving back gains due to market reversals
3. The daily profit target is reset each day at midnight
4. You can set your own target with the `--target` flag (e.g., `--target 20` for a $20 daily target)

Example usage with a $20 daily profit target:
```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 15 --target 20 --test
```

The bot tracks daily profits and reports how many days met the target in the trading results.

## Stop Loss and Take Profit

The bot uses two methods to protect your trades:

1. **Real Exchange Orders**: When opening a position, the bot places actual stop loss and take profit orders directly on Binance. These orders will execute immediately when the price reaches the specified levels, even if the bot is not running or checking at that moment.

2. **Daily Profit Target**: In addition to the take profit order, the bot also checks if your daily profit target has been reached during each update interval. If it has, it will close the position to secure your profits.

The stop loss and take profit levels are calculated dynamically based on the Average True Range (ATR) indicator:
- Stop loss: 1.5x ATR (capped at 3% of entry price)
- Take profit: 2x ATR (capped at 6% of entry price)

This ensures that your risk management is adapted to current market volatility.

## Troubleshooting

### Precision Error

If you encounter the error "Precision is over the maximum defined for this asset", it means the quantity precision is too high for the asset you're trading. This can happen when:

1. Your investment amount is too small
2. The risk per trade is too small
3. The asset price is too high

The bot now automatically handles precision requirements for each asset by:
1. Retrieving the correct precision requirements from Binance
2. Adjusting position sizes to meet minimum quantity requirements
3. Ensuring orders meet minimum notional value requirements
4. Automatically retrying with adjusted precision if an error occurs

If you still encounter precision errors, try these solutions:
- Increase your investment amount (use the `--investment` flag)
- Try a different trading pair with a lower price
- Run in test mode first to verify settings (`--test` flag)

### Minimum Position Size Requirements

Binance has minimum requirements for order sizes:

1. **Minimum Quantity**: Each trading pair has a minimum quantity (e.g., 0.001 for SOLUSDT)
2. **Minimum Notional Value**: The order value (quantity × price) must exceed a minimum amount (e.g., $10)

If your calculated position size is too small, you might see errors like:
- "Quantity less than or equal to zero"
- "Order value is below minimum notional"

The bot now automatically handles these requirements by:
1. Checking if the calculated position size meets minimum requirements
2. Adjusting the position size upward if needed
3. Using the minimum valid quantity as a fallback

For high-priced assets like SOLUSDT (>$150), we recommend:
- Using an investment amount of at least $100
- Setting leverage to 20x
- Using a risk per trade of at least 2%
- Alternatively, use the `--full-investment` flag to use your entire investment amount (with higher risk)

### Trading with Small Investment Amounts

If you have a small investment amount (e.g., $50), you have several options:

1. **Use Lower-Priced Assets**: Trading pairs like DOGEUSDT or XRPUSDT work better with small investments
2. **Use the Full Investment Mode**: Add the `--full-investment` flag to use your entire investment for each trade:
   ```
   python run_realtime_trading.py --symbol ADAUSDT --investment 50 --leverage 20 --full-investment --test
   ```
3. **Increase Leverage**: Use the maximum allowed leverage (20x) to maximize your trading power

**Note**: Using the `--full-investment` flag significantly increases risk as you're using your entire investment for each trade. Always test first with the `--test` flag before using real money.

### Other Common Issues

#### API Key Permissions
Make sure your Binance API key has the correct permissions:
- Read permissions for market data
- Trading permissions for executing orders
- If using futures, ensure futures trading is enabled for your API key

#### Network Issues
If you encounter connection errors, check your internet connection and Binance API status.

## Results and Reporting

Trading results are saved in the `realtime_results` directory:
- Trade history: `{symbol}_trade_history.csv` (or `test_{symbol}_trade_history.csv` for test mode)
- Current status: `{symbol}_status.csv` (or `test_{symbol}_status.csv` for test mode)

## License

This project is licensed under the MIT License - see the LICENSE file for details.