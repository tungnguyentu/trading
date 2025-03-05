#!/usr/bin/env python3
import argparse
from realtime import RealtimeSimulator

def main():
    parser = argparse.ArgumentParser(description='Real-time Trading Simulator')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair to simulate')
    parser.add_argument('--investment', type=float, default=50.0, help='Initial investment amount')
    parser.add_argument('--target', type=float, default=15.0, help='Daily profit target')
    parser.add_argument('--hours', type=int, default=24, help='Duration in hours')
    parser.add_argument('--interval', type=int, default=15, help='Update interval in minutes')
    parser.add_argument('--leverage', type=int, default=5, help='Margin trading leverage (1-20x)')
    
    args = parser.parse_args()
    
    simulator = RealtimeSimulator(
        symbol=args.symbol,
        initial_investment=args.investment,
        daily_profit_target=args.target,
        leverage=args.leverage
    )
    
    simulator.run_realtime_simulation(
        duration_hours=args.hours,
        update_interval_minutes=args.interval
    )

if __name__ == "__main__":
    main()