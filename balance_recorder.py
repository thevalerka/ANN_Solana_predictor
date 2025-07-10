"""
Balance Change Data Recorder - Record All Balance Changes to JSON

RECORDING LOGIC:
- Monitor balance_change_tracking.json every second
- Record ALL balance changes to output JSON file
- Format: time, coinname, increase/decrease, amount, %value, price_of_coin
- Track price data from WebSocket feeds
- No actual trading - data recording only
"""

import websocket
import json
import threading
import os
import time
from datetime import datetime
from collections import defaultdict

# Configuration
BALANCE_TRACKING_FILE = "/home/ubuntu/009_MM_BOTS/bot003_oppositeMDu/balance_change_tracking.json"
OUTPUT_DATA_FILE = "/home/ubuntu/009_MM_BOTS/bot004_NeuralNetwork/data/balance_change_records.json"  # Output file for recorded data
SCAN_INTERVAL = 5.0        # Scan balance file every 5 seconds

# Global storage for prices and recorded data
current_prices = {}  # {coin: {'bid': price, 'ask': price}}
last_processed_cycle = {}  # Store last processed cycle for each coin to avoid duplicates
recorded_data = []  # Store all recorded balance change events

# WebSocket connection
paradex_ws = None

# Coin configurations for price tracking
COIN_CONFIGS = {
    'WIF': {'market': 'WIF-USD-PERP'},
    'JUP': {'market': 'JUP-USD-PERP'},
    'PYTH': {'market': 'PYTH-USD-PERP'},
    'PENDLE': {'market': 'PENDLE-USD-PERP'},
    'MOODENG': {'market': 'RAY-USD-PERP'},
    'RAY': {'market': 'RAY-USD-PERP'},
    'SPX': {'market': 'SPX-USD-PERP'},
    'TRUMP': {'market': 'TRUMP-USD-PERP'},
    'AI16Z': {'market': 'AI16Z-USD-PERP'},
    'AIXBT': {'market': 'AIXBT-USD-PERP'},
    'VIRTUAL': {'market': 'VIRTUAL-USD-PERP'},
    'GOAT': {'market': 'GOAT-USD-PERP'},
    'POPCAT': {'market': 'POPCAT-USD-PERP'},
    'PNUT': {'market': 'PNUT-USD-PERP'},
    'NEIRO': {'market': 'NEIRO-USD-PERP'},
    'FARTCOIN': {'market': 'FARTCOIN-USD-PERP'},
    'MELANIA': {'market': 'MELANIA-USD-PERP'},
    'VINE': {'market': 'VINE-USD-PERP'},
    'PENGU': {'market': 'PENGU-USD-PERP'},
}

def load_balance_tracking():
    """Load balance tracking data from JSON file"""
    try:
        if os.path.exists(BALANCE_TRACKING_FILE):
            with open(BALANCE_TRACKING_FILE, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️  Balance tracking file not found: {BALANCE_TRACKING_FILE}")
            return {}
    except Exception as e:
        print(f"❌ Error loading balance tracking: {e}")
        return {}

def save_recorded_data():
    """Save recorded data to JSON file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(OUTPUT_DATA_FILE), exist_ok=True)
        
        with open(OUTPUT_DATA_FILE, 'w') as f:
            json.dump(recorded_data, f, indent=2)
        print(f"💾 Data saved to {OUTPUT_DATA_FILE} ({len(recorded_data)} records)")
    except Exception as e:
        print(f"❌ Error saving recorded data: {e}")

def get_current_mid_price(coin):
    """Get current mid price for a coin"""
    if coin not in current_prices:
        return None

    bid = current_prices[coin].get('bid', 0)
    ask = current_prices[coin].get('ask', 0)

    if bid <= 0 or ask <= 0:
        return None

    return (bid + ask) / 2

def record_balance_change(coin, change_type, consecutive_count, balance_change_amount, balance_change_percent, change_timestamp=None):
    """Record a balance change event to the data file"""
    if change_timestamp:
        current_time = datetime.fromtimestamp(change_timestamp).isoformat()
    else:
        current_time = datetime.now().isoformat()
    
    current_price = get_current_mid_price(coin)
    
    # Skip recording if price is invalid
    if current_price is None or current_price == 0.0:
        print(f"⚠️ SKIPPED RECORDING: {coin} {change_type} - No valid price data (price: {current_price})")
        return
    
    # Skip recording if amount or percent_value is 0.0
    amount = balance_change_amount if balance_change_amount else 0.0
    percent = balance_change_percent if balance_change_percent else 0.0
    
    if amount == 0.0 or percent == 0.0:
        print(f"⚠️ SKIPPED RECORDING: {coin} {change_type} - Zero amount ({amount:.4f}) or percent ({percent:.2f}%)")
        return
    
    record = {
        "time": current_time,
        "coinname": coin,
        "change_direction": change_type,  # "increase" or "decrease"
        "consecutive_count": consecutive_count,
        "amount": amount,
        "percent_value": percent,
        "price_of_coin": current_price
    }
    
    recorded_data.append(record)
    
    print(f"📝 RECORDED: {coin} {change_type} - Amount: {amount:.4f} ({percent:.4f}%) - Price: ${current_price:.6f}")
    
    # Save to file immediately
    save_recorded_data()

def analyze_balance_changes():
    """Analyze balance changes and record ALL new balance change events from change_history"""
    balance_data = load_balance_tracking()

    if not balance_data:
        print("⚠️ No balance tracking data found")
        return

    current_time = time.time()
    new_records_count = 0

    for coin, tracking_data in balance_data.items():
        if coin not in COIN_CONFIGS:
            continue  # Skip coins we don't support

        # Get the change history array
        change_history = tracking_data.get('change_history', [])
        if not change_history:
            continue

        # Get the last processed cycle for this coin
        last_cycle = last_processed_cycle.get(coin, 0)
        
        # Find new changes since last processed cycle
        new_changes = []
        max_cycle = last_cycle

        for change in change_history:
            change_cycle = change.get('cycle', 0)
            change_timestamp = change.get('timestamp', 0)
            
            # Only process changes newer than our last processed cycle
            # Also check timestamp to ensure it's within the last 5 seconds window
            time_diff = current_time - change_timestamp
            
            if change_cycle > last_cycle and time_diff <= 10:  # 10 second window for safety
                new_changes.append(change)
                max_cycle = max(max_cycle, change_cycle)

        # Process each new change
        for change in new_changes:
            direction = change.get('direction', '')
            change_amount = change.get('change_amount', 0.0)
            change_percentage = change.get('change_percentage', 0.0)
            change_cycle = change.get('cycle', 0)
            change_timestamp = change.get('timestamp', 0)
            
            # Convert timestamp to readable format for logging
            change_time = datetime.fromtimestamp(change_timestamp).strftime('%H:%M:%S')
            
            print(f"🔍 {coin} NEW CHANGE: {direction} {change_amount:.4f} ({change_percentage:.4f}%) at {change_time} cycle:{change_cycle}")
            
            # Record the change
            if direction == 'increase':
                record_balance_change(
                    coin=coin,
                    change_type="increase",
                    consecutive_count=1,  # Individual changes don't have consecutive count in this context
                    balance_change_amount=change_amount,
                    balance_change_percent=change_percentage,
                    change_timestamp=change_timestamp
                )
                new_records_count += 1
            elif direction == 'decrease':
                record_balance_change(
                    coin=coin,
                    change_type="decrease",
                    consecutive_count=1,
                    balance_change_amount=abs(change_amount),  # Make positive for recording
                    balance_change_percent=abs(change_percentage),  # Make positive for recording
                    change_timestamp=change_timestamp
                )
                new_records_count += 1

        # Update last processed cycle for this coin
        if max_cycle > last_cycle:
            last_processed_cycle[coin] = max_cycle
            print(f"📊 {coin} updated last processed cycle: {last_cycle} → {max_cycle}")

    if new_records_count > 0:
        print(f"✅ Processed {new_records_count} new balance changes across all coins")
    else:
        print("📊 No new balance changes found")

def balance_monitor():
    """Background thread to monitor and record all new balance changes from change_history"""
    print("📊 Balance monitor started - scanning change_history every 5 seconds")

    while True:
        try:
            analyze_balance_changes()
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            print(f"❌ Balance monitor error: {e}")
            time.sleep(SCAN_INTERVAL)

# WebSocket handlers
def on_paradex_message(ws, message):
    """Handle Paradex WebSocket messages for price updates"""
    try:
        data = json.loads(message)
        if data.get('method') == 'subscription' and 'params' in data:
            params = data['params']
            if 'data' in params:
                bbo = params['data']
                # Extract coin from channel name (e.g., "bbo.BTC-USD-PERP" -> "BTC")
                channel = params.get('channel', '')
                if channel.startswith('bbo.'):
                    market = channel.replace('bbo.', '')
                    coin = market.split('-')[0]

                    if coin in COIN_CONFIGS:
                        if coin not in current_prices:
                            current_prices[coin] = {}

                        bid = float(bbo.get('bid', 0))
                        ask = float(bbo.get('ask', 0))

                        current_prices[coin]['bid'] = bid
                        current_prices[coin]['ask'] = ask

    except Exception as e:
        pass  # Suppress WebSocket parsing errors

def on_paradex_open(ws):
    """Handle Paradex WebSocket connection opened"""
    print("✅ Connected to Paradex WebSocket")

    # Subscribe to all supported coins
    for coin, config in COIN_CONFIGS.items():
        subscription = {
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {"channel": f"bbo.{config['market']}"},
            "id": hash(coin) % 1000
        }
        ws.send(json.dumps(subscription))

    print(f"📡 Subscribed to {len(COIN_CONFIGS)} coin price feeds")

def on_paradex_error(ws, error):
    """Handle Paradex WebSocket errors"""
    print(f"❌ Paradex WebSocket error: {error}")

def on_paradex_close(ws, code, msg):
    """Handle Paradex WebSocket connection closed"""
    print(f"⚠️  Paradex WebSocket closed: {code} - {msg}")

def start_paradex_websocket():
    """Start Paradex WebSocket connection"""
    global paradex_ws

    paradex_ws = websocket.WebSocketApp(
        "wss://ws.api.prod.paradex.trade/v1",
        on_open=on_paradex_open,
        on_message=on_paradex_message,
        on_error=on_paradex_error,
        on_close=on_paradex_close
    )

    paradex_ws.run_forever()

def print_status():
    """Print current status and statistics"""
    while True:
        try:
            time.sleep(60)  # Print status every minute

            print(f"\n{'='*60}")
            print(f"📊 BALANCE CHANGE RECORDER STATUS - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            # Balance tracking status
            balance_data = load_balance_tracking()
            coins_with_activity = 0

            if balance_data:
                print(f"📈 Tracking {len(balance_data)} coins for balance changes")

                for coin, tracking in balance_data.items():
                    if coin not in COIN_CONFIGS:
                        continue

                    change_history = tracking.get('change_history', [])
                    last_cycle = last_processed_cycle.get(coin, 0)
                    latest_cycle = tracking.get('last_updated_cycle', 0)
                    total_changes = len(change_history)
                    
                    if total_changes > 0:
                        # Show recent activity info
                        recent_changes = [c for c in change_history if c.get('cycle', 0) > latest_cycle - 10]
                        recent_increases = len([c for c in recent_changes if c.get('direction') == 'increase'])
                        recent_decreases = len([c for c in recent_changes if c.get('direction') == 'decrease'])
                        
                        print(f"   {coin}: {total_changes} total changes, last cycle: {latest_cycle}, processed: {last_cycle} (recent: ↗️{recent_increases} ↘️{recent_decreases})")
                        coins_with_activity += 1

            if coins_with_activity == 0:
                print("   No coins with change history")
                
            # Last processed cycles status
            if last_processed_cycle:
                print(f"\n📊 Last Processed Cycles:")
                for coin, cycle in last_processed_cycle.items():
                    print(f"   {coin}: cycle {cycle}")
            else:
                print(f"\n📊 No cycles processed yet")

            # Price data status
            coins_with_prices = len([c for c in current_prices.values() if c.get('bid', 0) > 0])
            print(f"\n📡 Price feeds: {coins_with_prices}/{len(COIN_CONFIGS)} coins active")

            # Recording statistics
            total_records = len(recorded_data)
            if total_records > 0:
                recent_records = [r for r in recorded_data if (datetime.now() - datetime.fromisoformat(r['time'])).total_seconds() < 3600]
                print(f"\n📝 Recording Statistics:")
                print(f"   Total records: {total_records}")
                print(f"   Records in last hour: {len(recent_records)}")
                print(f"   Output file: {OUTPUT_DATA_FILE}")
                
                if recent_records:
                    increase_records = len([r for r in recent_records if r['change_direction'] == 'increase'])
                    decrease_records = len([r for r in recent_records if r['change_direction'] == 'decrease'])
                    print(f"   Last hour breakdown: {increase_records} increases, {decrease_records} decreases")

            print(f"{'='*60}")

        except Exception as e:
            print(f"❌ Status update error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Balance Change Data Recorder")
    print(f"📁 Monitoring: {BALANCE_TRACKING_FILE}")
    print(f"💾 Output file: {OUTPUT_DATA_FILE}")
    print(f"🔄 Scan interval: {SCAN_INTERVAL}s")

    print("\n📋 Recording Logic:")
    print("   • Scan balance_change_tracking.json every 5 seconds")
    print("   • Process change_history array for new entries")
    print("   • Track last processed cycle per coin to avoid duplicates")
    print("   • Record ALL new balance changes (increases and decreases)")
    print("   • Save to JSON: time, coinname, increase/decrease, amount, %value, price")
    print("   • Track real-time prices from WebSocket")
    print("   • No actual trading - data recording only")
    print("   • Skip records with zero price, amount, or percent values")

    print(f"\n🪙 Supported coins: {len(COIN_CONFIGS)}")
    print("-" * 60)

    # Load any existing recorded data
    if os.path.exists(OUTPUT_DATA_FILE):
        try:
            with open(OUTPUT_DATA_FILE, 'r') as f:
                recorded_data = json.load(f)
            print(f"📂 Loaded {len(recorded_data)} existing records from {OUTPUT_DATA_FILE}")
        except Exception as e:
            print(f"⚠️ Could not load existing data file: {e}")
            recorded_data = []
    else:
        recorded_data = []
        # Ensure directory exists
        os.makedirs(os.path.dirname(OUTPUT_DATA_FILE), exist_ok=True)
        print(f"📂 Creating new data file: {OUTPUT_DATA_FILE}")

    # Start background threads
    threading.Thread(target=balance_monitor, daemon=True).start()
    threading.Thread(target=print_status, daemon=True).start()

    # Start WebSocket connection
    threading.Thread(target=start_paradex_websocket, daemon=True).start()

    print("✅ All systems started")
    print("📊 Status reports every 60 seconds")
    print("📝 Recording balance change events automatically")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Balance Change Data Recorder...")

        # Final save
        save_recorded_data()

        # Final statistics
        total_records = len(recorded_data)
        if total_records > 0:
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total records saved: {total_records}")
            print(f"   Output file: {OUTPUT_DATA_FILE}")
            
            increase_records = len([r for r in recorded_data if r['change_direction'] == 'increase'])
            decrease_records = len([r for r in recorded_data if r['change_direction'] == 'decrease'])
            print(f"   Breakdown: {increase_records} increases, {decrease_records} decreases")
            
            # Show final processed cycles
            if last_processed_cycle:
                print(f"   Final processed cycles:")
                for coin, cycle in last_processed_cycle.items():
                    print(f"     {coin}: cycle {cycle}")

        print("👋 Recorder stopped successfully")
