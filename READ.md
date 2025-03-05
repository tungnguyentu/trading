# Complete Guide to All New Trading Bot Settings

## Enhanced Signal System
- `--enhanced-signals`: Activates multi-indicator confirmation system  
- `--signal-threshold [1-8]`: Sets how many indicators must agree before taking a trade  
  - Higher values (3-4) = More accurate but fewer trades  
  - Lower values (1-2) = More trades but potentially lower accuracy  
  - **Default:** 2  

## Trend Following
- `--trend-following`: Only trades in the direction of the overall market trend  
- Analyzes multiple timeframes to determine if the market is in an uptrend or downtrend  
- Rejects signals that go against the identified trend  
- Helps avoid fighting against strong market momentum  

## Pyramiding Strategy
- `--pyramiding`: Enables adding to winning positions  
- `--pyramid-entries [1-5]`: Maximum number of additional entries allowed  
  - Each entry adds to your position size as the trade moves in your favor  
  - More entries = More potential profit but also more exposure  
  - **Default:** 2  
- `--pyramid-threshold [0.5-5.0]`: Profit percentage required before adding to position  
  - Higher threshold = More conservative, only adds after significant profit  
  - Lower threshold = More aggressive, adds earlier in the trade  
  - **Default:** 1.0%  

## Dynamic Take Profit
- `--dynamic-tp`: Adjusts take profit targets based on market conditions  
  - **In strong trends**: Sets more ambitious targets  
  - **In choppy markets**: Sets more conservative targets  
- Uses volatility measurements to determine appropriate exit levels  
- Adapts to changing market conditions during the trade  

## Compound Interest
- `--compound`: Uses profits to increase position sizes over time  
- As your account grows, your position sizes grow proportionally  
- Accelerates profits during winning streaks  
- Can significantly increase long-term returns  

## Risk Management Options
- `--full-investment`: Uses full investment amount for each trade (**higher risk**)  
- `--full-margin`: Uses full investment as margin (**extreme risk, not recommended**)  

## Recommended Combinations

### Conservative Setup
- **Higher signal threshold (4)**  
- **Lower leverage (5x)**  
- **No pyramiding**  
- **Trend following for safety**  

### Balanced Setup
- **Moderate signal threshold (3)**  
- **Moderate leverage (10x)**  
- **Conservative pyramiding (1.5% threshold)**  
- **Compound interest for growth**  

### Aggressive Setup
- **Lower signal threshold (2)**  
- **Higher leverage (15x)**  
- **More aggressive pyramiding (3 entries, 1% threshold)**  
- **Dynamic take profit to maximize gains**  
- **No trend following (trades both with and against trends)**  

## How These Settings Work Together
These settings create a comprehensive trading system that:  
- **Finds quality entries** (*enhanced signals*)  
- **Aligns with market direction** (*trend following*)  
- **Maximizes winning trades** (*pyramiding*)  
- **Optimizes exits** (*dynamic take profit*)  
- **Grows account over time** (*compound interest*)  

Each setting can be adjusted independently to match your risk tolerance and trading style.  