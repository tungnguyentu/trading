# Complete Guide to All Trading Bot Settings

## Basic Parameters
- `--symbol`: Trading symbol (default: BTCUSDT)
- `--investment`: Initial investment amount in USDT (default: 50.0)
- `--profit-target`: Daily profit target percentage (default: 15.0)
- `--leverage`: Leverage for futures trading (default: 5)
- `--duration`: Trading duration in hours (default: 24)
- `--interval-minutes`: Update interval in minutes (default: 15)
- `--interval-seconds`: Additional update interval in seconds (default: 0)
- `--test-mode`: Run in test mode without real trades

## Enhanced Signal System
- `--enhanced-signals`: Activates multi-indicator confirmation system  
- `--signal-threshold [1-8]`: Sets how many indicators must agree before taking a trade  
  - Higher values (3-4) = More accurate but fewer trades  
  - Lower values (1-2) = More trades but potentially lower accuracy  
  - **Default:** 2  
- `--signal-cooldown [5-60]`: Minimum time between signals in minutes
  - Higher values = Fewer trades, less noise
  - Lower values = More trading opportunities
  - **Default:** 15

## Trend Following
- `--trend-following`: Only trades in the direction of the overall market trend  
- Analyzes multiple timeframes to determine if the market is in an uptrend or downtrend  
- Rejects signals that go against the identified trend  
- Helps avoid fighting against strong market momentum  

## Pyramiding Strategy
- `--pyramiding`: Enables adding to winning positions  
- `--max-pyramid [1-5]`: Maximum number of additional entries allowed  
  - Each entry adds to your position size as the trade moves in your favor  
  - More entries = More potential profit but also more exposure  
  - **Default:** 2  
- `--pyramid-threshold [0.01-5.0]`: Profit percentage required before adding to position  
  - Higher threshold = More conservative, only adds after significant profit  
  - Lower threshold = More aggressive, adds earlier in the trade  
  - **Default:** 0.01%  

## Take Profit and Stop Loss Settings
- `--dynamic-tp`: Adjusts take profit targets based on market conditions  
  - **In strong trends**: Sets more ambitious targets  
  - **In choppy markets**: Sets more conservative targets  
  - Uses volatility measurements to determine appropriate exit levels  
  - Adapts to changing market conditions during the trade  
- `--fixed-tp [0-100]`: Fixed take profit percentage (0 to disable)
- `--fixed-sl [0-100]`: Fixed stop loss percentage (0 to disable)

## Scalping Mode
- `--scalping`: Enable scalping mode for small range trading with quick profits
- `--scalping-tp [0.1-1.0]`: Scalping take profit factor (default: 0.5)
- `--scalping-sl [0.1-1.0]`: Scalping stop loss factor (default: 0.8)

## Machine Learning Integration
- `--ml-signals`: Use machine learning for signal generation
- `--ml-confidence [0.5-1.0]`: Confidence threshold for ML signals (default: 0.6)
- `--train-ml`: Train ML model before trading
- `--retrain-interval [0-24]`: Retrain ML model every N hours (0 = no retraining)

## Position Management
- `--reassess`: Periodically reassess positions based on changing signals

## Risk Management Options
- `--compound`: Uses profits to increase position sizes over time  
- `--full-investment`: Uses full investment amount for each trade (**higher risk**)  
- `--full-margin`: Uses full investment as margin (**extreme risk, not recommended**)  

## Recommended Combinations

### Conservative Setup
```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 5 --profit-target 10 \
--enhanced-signals --signal-threshold 4 --trend-following --test-mode
```
- **Higher signal threshold (4)**  
- **Lower leverage (5x)**  
- **No pyramiding**  
- **Trend following for safety**  

### Balanced Setup
```
python run_realtime_trading.py --symbol BTCUSDT --investment 100 --leverage 10 --profit-target 15 \
--enhanced-signals --signal-threshold 3 --trend-following --pyramiding \
--max-pyramid 2 --pyramid-threshold 1.5 --dynamic-tp --compound --test-mode
```
- **Moderate signal threshold (3)**  
- **Moderate leverage (10x)**  
- **Conservative pyramiding (1.5% threshold)**  
- **Compound interest for growth**  

### Aggressive Setup
```
python run_realtime_trading.py --symbol ETHUSDT --investment 100 --leverage 15 --profit-target 20 \
--enhanced-signals --signal-threshold 2 --pyramiding --max-pyramid 3 \
--pyramid-threshold 1.0 --dynamic-tp --compound --ml-signals --train-ml --test-mode
```
- **Lower signal threshold (2)**  
- **Higher leverage (15x)**  
- **More aggressive pyramiding (3 entries, 1% threshold)**  
- **Dynamic take profit to maximize gains**  
- **ML signals for advanced entry/exit**  

### Scalping Setup
```
python run_realtime_trading.py --symbol BTCUSDT --investment 50 --leverage 10 --profit-target 5 \
--scalping --scalping-tp 0.3 --scalping-sl 0.5 --interval-minutes 5 --test-mode
```
- **Scalping mode for quick profits**
- **Shorter update interval (5 minutes)**
- **Lower take profit targets**
- **Tighter stop losses**

## Technical Indicators Support

The bot now supports multiple technical indicator libraries:

1. **TA-Lib** (primary, if installed)
2. **pandas_ta** (fallback #1)
3. **ta** (fallback #2)
4. Pure Python implementations (final fallback)

This ensures the bot can run even if TA-Lib is not installed, which can be challenging on some platforms. For best performance, installing TA-Lib is recommended but not required.

## How These Settings Work Together
These settings create a comprehensive trading system that:  
- **Finds quality entries** (*enhanced signals, ML*)  
- **Aligns with market direction** (*trend following*)  
- **Maximizes winning trades** (*pyramiding*)  
- **Optimizes exits** (*dynamic take profit, scalping*)  
- **Grows account over time** (*compound interest*)  
- **Manages risk** (*fixed stop loss, position sizing*)

Each setting can be adjusted independently to match your risk tolerance and trading style.  