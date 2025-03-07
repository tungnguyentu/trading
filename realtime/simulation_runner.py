import time
import re
from datetime import datetime, timedelta
from simulation import TradingSimulator


def run_simulation(realtime_simulator, duration_hours=24, update_interval_minutes=15):
    """
    Run real-time simulation

    Args:
        realtime_simulator: The RealtimeSimulator instance
        duration_hours: How long to run the simulation in hours
        update_interval_minutes: How often to update in minutes
    """
    print(
        f"Starting real-time simulation for {realtime_simulator.symbol} for {duration_hours} hours"
    )
    print(f"Update interval: {update_interval_minutes} minutes")

    # Initialize trading simulator
    realtime_simulator.simulator = TradingSimulator(
        realtime_simulator.symbol,
        realtime_simulator.initial_investment,
        realtime_simulator.leverage,
    )

    # Calculate start and end times
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    # Send notification
    realtime_simulator.send_notification(
        f"🚀 REAL-TIME SIMULATION STARTED\n"
        f"Symbol: {realtime_simulator.symbol}\n"
        f"Initial Investment: ${realtime_simulator.initial_investment:.2f}\n"
        f"Leverage: {realtime_simulator.leverage}x\n"
        f"Duration: {duration_hours} hours\n"
        f"Update Interval: {update_interval_minutes} minutes\n"
        f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Main simulation loop
    while datetime.now() < end_time:
        try:
            current_time = datetime.now()
            print(f"\n=== Update: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            # Get latest data
            latest_df = realtime_simulator.get_latest_data(
                lookback_candles=2
            )  # Just get the latest candles

            if latest_df is None or len(latest_df) < 1:
                print("Error fetching latest data, will retry next interval")
                time.sleep(60)  # Wait a minute before retrying
                continue

            # Get the latest candle
            latest_candle = latest_df.iloc[-1]

            # Get current price
            current_price = float(latest_candle["close"])
            print(f"Current {realtime_simulator.symbol} price: ${current_price:.2f}")

            # Check for take profit / stop loss
            tp_sl_result = realtime_simulator.check_take_profit_stop_loss(
                current_price, latest_candle["timestamp"]
            )

            if tp_sl_result:
                print(tp_sl_result)

                # Send position closed notification
                if "TAKE PROFIT" in tp_sl_result or "STOP LOSS" in tp_sl_result:
                    # Extract position type and profit/loss from result
                    position_type = "LONG" if "LONG" in tp_sl_result else "SHORT"
                    reason = (
                        "TAKE PROFIT" if "TAKE PROFIT" in tp_sl_result else "STOP LOSS"
                    )

                    # Extract profit amount from the result string
                    profit_match = re.search(
                        r"(Profit|Loss): \$([0-9.-]+)", tp_sl_result
                    )
                    profit_amount = float(profit_match.group(2)) if profit_match else 0

                    # Create emoji based on profit/loss
                    emoji = "💰" if profit_amount > 0 else "🛑"

                    # Send notification for position closed
                    close_message = (
                        f"{emoji} POSITION CLOSED ({position_type})\n"
                        f"Symbol: {realtime_simulator.symbol}\n"
                        f"Reason: {reason}\n"
                        f"Exit Price: ${current_price:,.2f}\n"
                        f"{'Profit' if profit_amount > 0 else 'Loss'}: ${abs(profit_amount):,.2f}\n"
                        f"Balance: ${realtime_simulator.simulator.current_balance:,.2f}"
                    )
                    realtime_simulator.send_notification(close_message)
            else:
                # Get trading signals
                traditional_signal = latest_candle["signal"]

                # Get ML signal with error handling
                ml_signal = 0
                ml_confidence = 0
                try:
                    if realtime_simulator.ml_manager:
                        ml_signal, ml_confidence = (
                            realtime_simulator.ml_manager.get_ml_signal(
                                realtime_simulator.symbol, latest_df
                            )
                        )
                except Exception as e:
                    print(f"Error getting ML signal: {e}")
                    ml_signal = 0
                    ml_confidence = 0

                # Combine signals
                signal = 0
                if traditional_signal == ml_signal:
                    signal = traditional_signal
                elif ml_confidence > 0.75:
                    signal = ml_signal
                else:
                    signal = traditional_signal  # Fallback to traditional signal

                # Execute trade if there's a signal
                if signal != 0:
                    # Execute the trade with risk management
                    trade_result = realtime_simulator.execute_trade(
                        signal, current_price, latest_candle["timestamp"]
                    )

                    if trade_result:
                        print(trade_result)

                        # Calculate stop loss and take profit levels for notification
                        if signal == 1:  # BUY signal
                            stop_loss_price = current_price * (
                                1 - realtime_simulator.simulator.stop_loss_pct
                            )
                            take_profit_price = current_price * (
                                1 + realtime_simulator.simulator.take_profit_pct
                            )
                        else:  # SELL signal
                            stop_loss_price = current_price * (
                                1 + realtime_simulator.simulator.stop_loss_pct
                            )
                            take_profit_price = current_price * (
                                1 - realtime_simulator.simulator.take_profit_pct
                            )

                        # Send notification for position opened
                        if "BUY" in trade_result or "SELL" in trade_result:
                            position_type = "LONG" if "BUY" in trade_result else "SHORT"
                            emoji = "🟢" if position_type == "LONG" else "🔴"

                            # Extract position size from trade result
                            size_match = re.search(r"([0-9.]+) units", trade_result)
                            position_size = (
                                float(size_match.group(1)) if size_match else 0
                            )

                            open_message = (
                                f"{emoji} POSITION OPENED ({position_type})\n"
                                f"Symbol: {realtime_simulator.symbol}\n"
                                f"Entry Price: ${current_price:,.2f}\n"
                                f"Position Size: {position_size:,.6f} units\n"
                                f"Stop Loss: ${stop_loss_price:,.2f}\n"
                                f"Take Profit: ${take_profit_price:,.2f}\n"
                                f"Balance: ${realtime_simulator.simulator.current_balance:,.2f}"
                            )
                            realtime_simulator.send_notification(open_message)

            # Print current status
            print(
                f"Current balance: ${realtime_simulator.simulator.current_balance:.2f}"
            )
            if realtime_simulator.simulator.position:
                position_value = (
                    realtime_simulator.simulator.position_size * current_price
                )
                profit_loss = 0
                if realtime_simulator.simulator.position == "long":
                    profit_loss = realtime_simulator.simulator.position_size * (
                        current_price - realtime_simulator.simulator.entry_price
                    )
                else:  # short
                    profit_loss = realtime_simulator.simulator.position_size * (
                        realtime_simulator.simulator.entry_price - current_price
                    )

                print(
                    f"Current position: {realtime_simulator.simulator.position.upper()}"
                )
                print(f"Entry price: ${realtime_simulator.simulator.entry_price:.2f}")
                print(
                    f"Position size: {realtime_simulator.simulator.position_size:.6f} units"
                )
                print(f"Position value: ${position_value:.2f}")
                print(f"Unrealized P/L: ${profit_loss:.2f}")

                # Send position update notification on every update
                if realtime_simulator.simulator.position:
                    # Calculate profit percentage
                    profit_pct = 0
                    if realtime_simulator.simulator.position == "long":
                        profit_pct = (
                            (current_price - realtime_simulator.simulator.entry_price)
                            / realtime_simulator.simulator.entry_price
                            * 100
                        )
                    else:  # short
                        profit_pct = (
                            (realtime_simulator.simulator.entry_price - current_price)
                            / realtime_simulator.simulator.entry_price
                            * 100
                        )

                    # Only send update if significant change (>1% profit change)
                    if abs(profit_pct) > 1:
                        emoji = "📈" if profit_pct > 0 else "📉"
                        update_message = (
                            f"{emoji} POSITION UPDATE ({realtime_simulator.simulator.position.upper()})\n"
                            f"Symbol: {realtime_simulator.symbol}\n"
                            f"Current Price: ${current_price:,.2f}\n"
                            f"Entry Price: ${realtime_simulator.simulator.entry_price:,.2f}\n"
                            f"Unrealized P/L: ${profit_loss:,.2f} ({profit_pct:.2f}%)\n"
                            f"Stop Loss: ${realtime_simulator.simulator.stop_loss_price:,.2f}\n"
                            f"Take Profit: ${realtime_simulator.simulator.take_profit_price:,.2f}"
                        )
                        realtime_simulator.send_notification(update_message)

            # Save results
            realtime_simulator.save_realtime_results()

            # Wait for next update with countdown
            sleep_seconds = update_interval_minutes * 60
            print(f"Next update in {update_interval_minutes} minutes...")
            print(f"Next update in {sleep_seconds} seconds")
            print("Countdown started...")

            # Display countdown timer
            start_wait = time.time()
            total_wait = sleep_seconds

            while time.time() - start_wait < sleep_seconds:
                try:
                    # Calculate elapsed and remaining time
                    elapsed = time.time() - start_wait
                    remaining = sleep_seconds - elapsed
                    mins, secs = divmod(int(remaining), 60)

                    # Calculate progress percentage
                    progress_pct = elapsed / total_wait

                    # Create progress bar (width between 15-20 characters)
                    bar_width = 20
                    filled_width = int(bar_width * progress_pct)
                    bar = "█" * filled_width + "░" * (bar_width - filled_width)

                    # Create countdown display with margin
                    countdown = f"⏱️ Next update in: {mins:02d}:{secs:02d} [{bar}] {progress_pct:.0%}"
                    print(countdown, end="\r", flush=True)
                    time.sleep(1)

                except KeyboardInterrupt:
                    print("\nSimulation interrupted by user")
                    break

            print(
                "\nUpdate time reached!                                      "
            )  # Clear line with newline
            print(" " * 50, end="\r")  # Clear the line with extra space

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            break

        except Exception as e:
            print(f"Error in simulation loop: {e}")
            realtime_simulator.send_notification(f"⚠️ ERROR: {e}")
            # Wait a bit before retrying
            time.sleep(60)

    # Simulation completed
    print("\n=== Simulation Completed ===")
    print(f"Final balance: ${realtime_simulator.simulator.current_balance:.2f}")
    print(
        f"Profit/Loss: ${realtime_simulator.simulator.current_balance - realtime_simulator.initial_investment:.2f}"
    )
    print(
        f"Return: {((realtime_simulator.simulator.current_balance - realtime_simulator.initial_investment) / realtime_simulator.initial_investment) * 100:.2f}%"
    )

    # Close any open positions
    if realtime_simulator.simulator.position:
        print("Closing open position...")
        current_price = float(
            realtime_simulator.client.get_symbol_ticker(
                symbol=realtime_simulator.symbol
            )["price"]
        )
        realtime_simulator.simulator.close_position(
            current_price, datetime.now(), "simulation_end"
        )

    # Save final results
    realtime_simulator.save_realtime_results()

    # Send completion notification
    realtime_simulator.send_notification(
        f"🏁 SIMULATION COMPLETED\n"
        f"Symbol: {realtime_simulator.symbol}\n"
        f"Final Balance: ${realtime_simulator.simulator.current_balance:.2f}\n"
        f"Profit/Loss: ${realtime_simulator.simulator.current_balance - realtime_simulator.initial_investment:.2f}\n"
        f"Return: {((realtime_simulator.simulator.current_balance - realtime_simulator.initial_investment) / realtime_simulator.initial_investment) * 100:.2f}%\n"
        f"Total Trades: {len(realtime_simulator.simulator.trades)}"
    )

    return {
        "final_balance": realtime_simulator.simulator.current_balance,
        "profit_loss": realtime_simulator.simulator.current_balance
        - realtime_simulator.initial_investment,
        "return_pct": (
            (
                realtime_simulator.simulator.current_balance
                - realtime_simulator.initial_investment
            )
            / realtime_simulator.initial_investment
        )
        * 100,
        "total_trades": len(realtime_simulator.simulator.trades),
    }
