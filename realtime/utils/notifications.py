import telegram
import asyncio

def send_telegram_notification(bot_token, chat_id, message):
    """Send Telegram notification"""
    try:
        # Create a new event loop for each notification
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Define the async function
        async def send_message():
            bot = telegram.Bot(token=bot_token)
            await bot.send_message(chat_id=chat_id, text=message)
        
        # Run the coroutine and close the loop properly
        loop.run_until_complete(send_message())
        loop.close()
        print(f"Telegram notification sent: {message}")
    except Exception as e:
        print(f"Error sending Telegram notification: {e}")