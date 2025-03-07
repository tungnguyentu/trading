import time
from datetime import datetime, timedelta

def display_realtime_countdown(seconds, message="Next update in"):
    """
    Display a real-time countdown with progress bar that updates every second
    
    Args:
        seconds: Total seconds to count down
        message: Message to display before the countdown
    """
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=seconds)
    
    print(f"{message} {seconds:.0f} seconds")
    print("Countdown started...")
    
    try:
        while True:
            now = datetime.now()
            remaining = (end_time - now).total_seconds()
            
            if remaining <= 0:
                break
                
            # Format time
            mins, secs = divmod(int(remaining), 60)
            hours, mins = divmod(mins, 60)
            
            # Create progress bar
            progress_pct = (seconds - remaining) / seconds
            bar_length = 20
            filled_length = int(bar_length * progress_pct)
            bar = "■" * filled_length + "□" * (bar_length - filled_length)
            
            # Display countdown
            if hours > 0:
                countdown = f"⏱️ {message}: {hours:02d}:{mins:02d}:{secs:02d} [{bar}] {progress_pct:.0%}"
            else:
                countdown = f"⏱️ {message}: {mins:02d}:{secs:02d} [{bar}] {progress_pct:.0%}"
                
            print(countdown, end='\r', flush=True)
            time.sleep(1)
            
        print("\nUpdate time reached!                                      ")
        print(" " * 50, end='\r')
        return True
        
    except KeyboardInterrupt:
        print("\nCountdown interrupted by user")
        return False
    except Exception as e:
        print(f"\nError in countdown: {e}")
        return False
