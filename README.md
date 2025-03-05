# Advanced Cryptocurrency Trading Bot

This trading bot implements multiple advanced trading strategies for cryptocurrency markets, combining technical analysis, risk management, and position optimization techniques.

## Features

- Real-time trading on Binance
- Multiple technical indicators for signal generation
- Enhanced signal confirmation system
- Trend following capabilities
- Pyramiding strategy for adding to winning positions
- Dynamic take profit targets
- Compound interest mode
- Configurable risk management
- Test mode with simulated balance

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Binance API keys as environment variables:
   ```
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_API_SECRET="your_api_secret"
   ```

## Usage

Run the bot with the following command:

```bash
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 10 [OPTIONS]
```

### Basic Parameters

- `--symbol`: Trading pair (e.g., BTCUSDT, ETHUSDT, ADAUSDT)
- `--investment`: Amount to invest per trade in USD
- `--leverage`: Trading leverage (1-20)
- `--target`: Daily profit target in USD (not percentage)
- `--hours`: Duration to run the bot in hours
- `--interval`: Update interval in minutes
- `--test`: Run in test mode with simulated balance

### Advanced Strategy Parameters

#### Enhanced Signal System
- `--enhanced-signals`: Activates multi-indicator confirmation system
- `--signal-threshold [1-8]`: Sets how many indicators must agree before taking a trade
  - Higher values (3-4) = More accurate but fewer trades
  - Lower values (1-2) = More trades but potentially lower accuracy
  - Default: 2

#### Trend Following
- `--trend-following`: Only trades in the direction of the overall market trend
  - Analyzes multiple timeframes to determine if the market is in an uptrend or downtrend
  - Rejects signals that go against the identified trend
  - Helps avoid fighting against strong market momentum

#### Pyramiding Strategy
- `--pyramiding`: Enables adding to winning positions
- `--pyramid-entries [1-5]`: Maximum number of additional entries allowed
  - Each entry adds to your position size as the trade moves in your favor
  - More entries = More potential profit but also more exposure
  - Default: 2
- `--pyramid-threshold [0.5-5.0]`: Profit percentage required before adding to position
  - Higher threshold = More conservative, only adds after significant profit
  - Lower threshold = More aggressive, adds earlier in the trade
  - Default: 1.0%

#### Dynamic Take Profit
- `--dynamic-tp`: Adjusts take profit targets based on market conditions
  - In strong trends: Sets more ambitious targets
  - In choppy markets: Sets more conservative targets
  - Uses volatility measurements to determine appropriate exit levels
  - Adapts to changing market conditions during the trade

#### Risk Management Options
- `--compound`: Uses profits to increase position sizes over time
- `--full-investment`: Uses full investment amount for each trade (higher risk)
- `--full-margin`: Uses full investment as margin (extreme risk, not recommended)

## Recommended Setups

### Best Overall Setup (Balanced Risk/Reward)
```bash
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 10 --target 5 --hours 24 --interval 5 --enhanced-signals --signal-threshold 3 --pyramiding --pyramid-entries 2 --pyramid-threshold 1.5 --trend-following --compound --test
```

### For Beginners (Conservative)
```bash
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 5 --target 3 --hours 24 --interval 5 --enhanced-signals --signal-threshold 4 --trend-following --test
```

### For Experienced Traders (Aggressive)
```bash
python run_realtime_trading.py --symbol ETHUSDT --investment 100 --leverage 15 --target 8 --hours 24 --interval 3 --enhanced-signals --signal-threshold 2 --pyramiding --pyramid-entries 3 --pyramid-threshold 1.0 --dynamic-tp --compound --test
```

### For Volatile Altcoins
```bash
python run_realtime_trading.py --symbol SOLUSDT --investment 50 --leverage 10 --target 10 --hours 24 --interval 5 --enhanced-signals --signal-threshold 3 --pyramiding --pyramid-entries 2 --pyramid-threshold 2.0 --trend-following --dynamic-tp --test
```

### For Stable Market Conditions
```bash
python run_realtime_trading.py --symbol BTCUSDT --investment 100 --leverage 5 --target 3 --hours 48 --interval 10 --enhanced-signals --signal-threshold 2 --pyramiding --pyramid-entries 3 --pyramid-threshold 1.0 --compound --test
```

## Risk Warning

Trading cryptocurrencies involves significant risk and can result in the loss of your invested capital. Always start with small amounts and test thoroughly before committing larger sums. The bot includes a test mode (`--test` flag) that simulates trading without using real funds.

## License

This project is licensed under the MIT License - see the LICENSE file for details.