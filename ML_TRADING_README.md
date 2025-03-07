# ML-Based Real-Time Trading

This system integrates machine learning models with real-time trading to generate trading signals based on ML predictions.

## Overview

The ML trading system combines two key components:
1. **ML Training and Backtesting**: Trains ML models on historical data and evaluates their performance
2. **Real-Time Trading**: Uses the trained models to generate trading signals and execute trades

## Features

- Train ML models on historical cryptocurrency data
- Backtest models to evaluate performance before live trading
- Use trained models to generate real-time trading signals
- Execute trades based on ML predictions
- Configurable ML confidence thresholds
- Periodic model retraining
- Test mode for paper trading
- Customizable take profit and stop loss settings

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- python-binance
- python-dotenv

## Setup

1. Make sure you have a `.env` file in the root directory with your Binance API keys:

```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

2. Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib python-binance python-dotenv
```

## Usage

### Basic ML Trading

To start trading with ML signals using default settings:

```bash
python run_ml_trading.py --symbol BTCUSDT
```

### Training and Backtesting Before Trading

To train a model, run a backtest, and then start trading if the results are good:

```bash
python run_ml_trading.py --symbol BTCUSDT --backtest-first --optimize
```

### Test Mode (Paper Trading)

To run in test mode without executing real trades:

```bash
python run_ml_trading.py --symbol BTCUSDT --test-mode
```

### Customizing ML Parameters

Customize the ML model and training parameters:

```bash
python run_ml_trading.py --symbol BTCUSDT --model-type ensemble --timeframe 1h --lookback-days 365 --optimize --ml-confidence 0.7
```

### Periodic Model Retraining

Set up periodic retraining of the ML model during trading:

```bash
python run_ml_trading.py --symbol BTCUSDT --retrain-interval 12
```

### Take Profit and Stop Loss Settings

Configure take profit and stop loss settings:

```bash
python run_ml_trading.py --symbol BTCUSDT --fixed-tp 3 --fixed-sl 2
```

Or use dynamic take profit based on market conditions:

```bash
python run_ml_trading.py --symbol BTCUSDT --dynamic-tp
```

## Command Line Arguments

```
usage: run_ml_trading.py [-h] [--symbol SYMBOL] [--investment INVESTMENT]
                         [--profit-target PROFIT_TARGET] [--leverage LEVERAGE]
                         [--duration DURATION] [--interval-minutes INTERVAL_MINUTES]
                         [--model-type {rf,gb,ensemble}] [--timeframe {1m,5m,15m,30m,1h,4h,1d}]
                         [--lookback-days LOOKBACK_DAYS] [--optimize]
                         [--force-retrain] [--ml-confidence ML_CONFIDENCE]
                         [--retrain-interval RETRAIN_INTERVAL] [--test-mode]
                         [--backtest-first] [--compound] [--dynamic-tp]
                         [--fixed-tp FIXED_TP] [--fixed-sl FIXED_SL]

Run ML-based real-time trading

optional arguments:
  -h, --help            show this help message and exit
  --symbol SYMBOL       Trading symbol
  --investment INVESTMENT
                        Initial investment amount in USDT
  --profit-target PROFIT_TARGET
                        Daily profit target percentage
  --leverage LEVERAGE   Leverage for futures trading
  --duration DURATION   Trading duration in hours
  --interval-minutes INTERVAL_MINUTES
                        Update interval in minutes
  --model-type {rf,gb,ensemble}
                        ML model type (rf=Random Forest, gb=Gradient Boosting,
                        ensemble=Ensemble)
  --timeframe {1m,5m,15m,30m,1h,4h,1d}
                        Timeframe for ML model training
  --lookback-days LOOKBACK_DAYS
                        Number of days of historical data to use for training
  --optimize            Optimize ML model hyperparameters
  --force-retrain       Force retraining of ML model even if it exists
  --ml-confidence ML_CONFIDENCE
                        ML confidence threshold for trading signals
  --retrain-interval RETRAIN_INTERVAL
                        ML model retraining interval in hours (0 to disable)
  --test-mode           Run in test mode without real trades
  --backtest-first      Run backtest before live trading
  --compound            Enable compound interest
  --dynamic-tp          Use dynamic take profit based on market conditions
  --fixed-tp FIXED_TP   Fixed take profit percentage (0 to disable)
  --fixed-sl FIXED_SL   Fixed stop loss percentage (0 to disable)
```

## Example Output

When running the ML trading script with backtesting, you'll see output like this:

```
=== Training and Backtesting ML Model ===
Symbol: BTCUSDT
Timeframe: 1h
Model Type: ensemble
Lookback Period: 365 days
Optimize Hyperparameters: True
Force Retrain: True

Fetching historical data for BTCUSDT (1h)...
Fetched 8760 candles for BTCUSDT

--- Training Advanced ML model for BTCUSDT ---
Data shape after feature engineering: (8712, 87)
Using 82 features
Training set size: 6969, Test set size: 1743
Training Ensemble model...
Optimizing hyperparameters...

=== Backtest Results ===
Total Return: 84.33%
Win Rate: 62.20%
Profit Factor: 2.18
Max Drawdown: 15.42%
Sharpe Ratio: 1.87

Proceed with live trading based on these results? (y/n): y

=== Trading Configuration ===
Symbol: BTCUSDT
Initial Investment: 50.0 USDT
Leverage: 5x
Duration: 24 hours
Update Interval: 15m
ML Confidence Threshold: 0.6

Enabled Modes: ML Signals, ML Training

Start trading with these settings? (y/n): y

Starting real-time trading for BTCUSDT...
```

## How It Works

1. **Training Phase**:
   - Historical data is fetched from Binance
   - Features are calculated (technical indicators, price patterns, etc.)
   - ML model is trained on the data
   - Model is backtested to evaluate performance

2. **Trading Phase**:
   - Real-time market data is fetched at regular intervals
   - Data is processed and features are calculated
   - ML model generates predictions with confidence scores
   - If confidence exceeds threshold, a trading signal is generated
   - Trades are executed based on the signals
   - Positions are managed with take profit and stop loss settings

3. **Retraining Phase** (if enabled):
   - Model is periodically retrained with new data
   - This helps adapt to changing market conditions

## Advanced Usage

### Programmatic Usage

You can also use the ML trading system programmatically in your own scripts:

```python
from realtime.trainer.ml_trainer import MLTrainer
from realtime.trader import RealtimeTrader

# Train and backtest model
trainer = MLTrainer(symbols=["BTCUSDT"], timeframe="1h", lookback_days=365)
backtest_results = trainer.train_and_backtest("BTCUSDT", model_type="ensemble", optimize_hyperparams=True)

# Check backtest results
if backtest_results and backtest_results["profit_factor"] > 1.5:
    # Create trader and start trading
    trader = RealtimeTrader(
        symbol="BTCUSDT",
        initial_investment=50.0,
        leverage=5,
        use_ml_signals=True,
        ml_confidence=0.6
    )
    
    # Run trading
    trader.run_real_trading(duration_hours=24, update_interval_minutes=15)
```

## Tips for Best Results

1. **Always backtest first**: Use the `--backtest-first` flag to evaluate model performance before live trading.

2. **Start with test mode**: Use `--test-mode` to run paper trading before risking real money.

3. **Optimize hyperparameters**: Use the `--optimize` flag to find the best model parameters.

4. **Adjust confidence threshold**: The `--ml-confidence` parameter controls how selective the system is with signals. Higher values mean fewer but potentially more reliable signals.

5. **Use appropriate timeframes**: Different symbols may perform better with different timeframes. Use the training system to compare performance across timeframes.

6. **Set reasonable take profit and stop loss**: Either use dynamic take profit (`--dynamic-tp`) or set fixed values that match the volatility of the asset.

7. **Monitor and adjust**: Regularly check the performance of your trading system and adjust parameters as needed. 