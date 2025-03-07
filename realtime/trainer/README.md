# ML Model Training and Backtesting

This module provides tools for training machine learning models on historical cryptocurrency data and backtesting their performance.

## Features

- Train ML models on historical cryptocurrency data
- Backtest models to evaluate performance
- Compare model performance across different timeframes
- Optimize hyperparameters for better performance
- Support for multiple model types (Random Forest, Gradient Boosting, Ensemble)
- Run real-time simulations with trained models

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

### Basic Training and Backtesting

To train models for BTC and ETH with default settings:

```bash
python train_and_backtest.py
```

### Customizing Symbols and Timeframes

Train models for specific symbols and timeframes:

```bash
python train_and_backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 4h
```

### Model Options

Choose different model types and enable hyperparameter optimization:

```bash
python train_and_backtest.py --model rf --optimize
```

Available model types:
- `rf`: Random Forest
- `gb`: Gradient Boosting
- `ensemble`: Ensemble of models (default)

### Comparing Timeframes

Compare model performance across different timeframes:

```bash
python train_and_backtest.py --symbols BTCUSDT --compare-timeframes
```

### Force Retraining

Force retraining even if models already exist:

```bash
python train_and_backtest.py --force
```

### Backtesting Options

Customize backtesting parameters:

```bash
python train_and_backtest.py --initial-balance 5000 --position-size 0.1
```

## Running Simulations

After training a model, you can run a real-time simulation to see how it performs:

```bash
python run_simulation.py --symbol BTCUSDT --timeframe 1h
```

### Training Before Simulation

You can also train a model right before running the simulation:

```bash
python run_simulation.py --symbol BTCUSDT --train-first --model-type ensemble --optimize
```

### Simulation Options

Customize the simulation parameters:

```bash
python run_simulation.py --symbol ETHUSDT --duration 48 --update-interval 30 --initial-investment 2000 --leverage 2
```

### Simulation Command Reference

```
usage: run_simulation.py [-h] [--symbol SYMBOL]
                         [--timeframe {1m,5m,15m,30m,1h,4h,1d}]
                         [--duration DURATION]
                         [--update-interval UPDATE_INTERVAL]
                         [--initial-investment INITIAL_INVESTMENT]
                         [--leverage LEVERAGE] [--profit-target PROFIT_TARGET]
                         [--train-first] [--model-type {rf,gb,ensemble}]
                         [--optimize]

Run a simulation with a trained ML model

optional arguments:
  -h, --help            show this help message and exit
  --symbol SYMBOL       Symbol to run simulation for
  --timeframe {1m,5m,15m,30m,1h,4h,1d}
                        Timeframe for simulation
  --duration DURATION   Duration of simulation in hours
  --update-interval UPDATE_INTERVAL
                        Update interval in minutes
  --initial-investment INITIAL_INVESTMENT
                        Initial investment amount in USD
  --leverage LEVERAGE   Leverage to use (1-20)
  --profit-target PROFIT_TARGET
                        Daily profit target in USD
  --train-first         Train model before running simulation
  --model-type {rf,gb,ensemble}
                        Type of model to train
  --optimize            Optimize hyperparameters during training
```

## Full Command Reference for Training and Backtesting

```
usage: train_and_backtest.py [-h] [--symbols SYMBOLS [SYMBOLS ...]]
                             [--timeframe {1m,5m,15m,30m,1h,4h,1d}]
                             [--lookback LOOKBACK] [--model {rf,gb,ensemble}]
                             [--force] [--optimize] [--compare-timeframes]
                             [--initial-balance INITIAL_BALANCE]
                             [--position-size POSITION_SIZE]

Train and backtest ML models for cryptocurrency trading

optional arguments:
  -h, --help            show this help message and exit
  --symbols SYMBOLS [SYMBOLS ...]
                        Symbols to train models for
  --timeframe {1m,5m,15m,30m,1h,4h,1d}
                        Timeframe for historical data
  --lookback LOOKBACK   Number of days of historical data to use
  --model {rf,gb,ensemble}
                        Type of model to train (rf=Random Forest, gb=Gradient
                        Boosting, ensemble=Ensemble of models)
  --force               Force retraining even if model exists
  --optimize            Optimize hyperparameters (takes longer but may improve
                        performance)
  --compare-timeframes  Compare performance across different timeframes
  --initial-balance INITIAL_BALANCE
                        Initial balance for backtesting
  --position-size POSITION_SIZE
                        Position size as percentage of balance (0.2 = 20%)
```

## Example Output

When running the training and backtesting, you'll see output like this:

```
=== ML Model Training and Backtesting ===
Symbols: BTCUSDT, ETHUSDT
Timeframe: 1h
Lookback period: 365 days
Model type: ensemble
Optimize hyperparameters: True
Force retrain: False
Initial balance: $10000.0
Position size: 20.0%

=== Training and Backtesting for BTCUSDT ===
Fetching historical data for BTCUSDT (1h)...
Fetched 8760 candles for BTCUSDT

--- Training Advanced ML model for BTCUSDT ---
Data shape after feature engineering: (8712, 87)
Using 82 features
Training set size: 6969, Test set size: 1743
Training Ensemble model...
Optimizing hyperparameters...

=== Backtest Results ===
Symbol: BTCUSDT
Model Type: ensemble
Initial Balance: $10000.00
Final Balance: $18432.56
Total Return: 84.33%
Annualized Return: 23.15%
Total Trades: 127
Win Rate: 62.20%
Profit Factor: 2.18
Max Drawdown: 15.42%
Sharpe Ratio: 1.87

=== Summary of All Results ===
Symbol | Return % | Win Rate | Profit Factor | Max Drawdown | Sharpe
-------|----------|----------|--------------|--------------|-------
BTCUSDT |   84.33% |  62.20% |         2.18 |       15.42% |   1.87
ETHUSDT |   97.65% |  64.81% |         2.43 |       18.76% |   2.12
```

## Simulation Output Example

When running a simulation, you'll see output like this:

```
=== ML Model Simulation ===
Symbol: BTCUSDT
Timeframe: 1h
Duration: 24 hours
Update interval: 15 minutes
Initial investment: $1000.0
Leverage: 1.0x
Daily profit target: $15.0

=== Starting Simulation ===
Starting real-time simulation for BTCUSDT for 24 hours
Update interval: 15 minutes

=== Update: 2023-06-01 12:00:00 ===
Current price: $27,345.67
ML Signal: BUY
Executing trade: BUY 0.0366 BTC at $27,345.67

=== Update: 2023-06-01 12:15:00 ===
Current price: $27,389.21
Position: LONG
Unrealized P&L: $1.59 (0.16%)

...

=== Simulation Complete ===
Final balance: $1087.32
Total return: 8.73%
Total trades: 7
Win rate: 57.14%
Equity curve saved to: equity_curve_BTCUSDT_20230601_120000.png
```

## Advanced Usage

### Using the MLTrainer Class in Your Code

You can also use the `MLTrainer` class directly in your Python code:

```python
from ml_trainer import MLTrainer

# Initialize trainer
trainer = MLTrainer(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframe='1h',
    lookback_days=365
)

# Train and backtest a model
results = trainer.train_and_backtest(
    symbol='BTCUSDT',
    model_type='ensemble',
    force_retrain=True,
    optimize_hyperparams=True
)

# Print results
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

### Running a Simulation Programmatically

You can also run a simulation programmatically:

```python
from realtime.simulator import RealtimeSimulator
from realtime.simulation_runner import run_simulation

# Initialize simulator
simulator = RealtimeSimulator(
    symbol='BTCUSDT',
    initial_investment=1000,
    daily_profit_target=15,
    leverage=1
)

# Run simulation
run_simulation(
    simulator,
    duration_hours=24,
    update_interval_minutes=15
)

# Print results
print(f"Final balance: ${simulator.simulator.balance:.2f}")
print(f"Total return: {(simulator.simulator.balance / 1000 - 1) * 100:.2f}%")