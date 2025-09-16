#!/usr/bin/env python3
"""
Test Strategy for Poloniex BTC Trading
Checks BTC price and places buy order if price < $100k USD
"""

##Test Paper Trade


import os
import sys
import time
import json
import pandas as pd
import numpy as np
from decimal import Decimal

# Add the parent directory to the path to import our modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, PROJECT_ROOT)

from constants import set_constants, get_constants
from exchange_api_spot.user import get_client_exchange
from utils import (
    get_line_number,
    update_key_and_insert_error_log,
    generate_random_string,
    get_precision_from_real_number,
    get_arg
)
## General log ##
from logger import logger_access, logger_error
from logger import setup_logger_global


class SimpleStrategy:
    """
    CCI Strategy:
    - Buy when CCI crosses above +100
    - Sell when CCI crosses below -100
    Spot mode: SELL là chốt coin đang có (không mở short).
    """

    def __init__(self, api_key="", secret_key="", passphrase="", session_key=""):
        # Identifiers
        self.run_key = generate_random_string()
        self.session_key = session_key

        # Strategy params
        self.symbol = "ETH"
        self.quote = "USDT"
        self.trade_value = 100              # Trade value in quote currency

        # === CCI params ===
        self.cci_period = 20
        self.cci_upper = 100
        self.cci_lower = -100

        self.interval = "1h"
        # Lấy nhiều hơn 1 chút để tính rolling/mad ổn định
        self.limit = max(self.cci_period * 4, 150)
        self.sleep = 60

        # Position tracking
        # 1 = đang long/đã mua, -1 = đã bán/hết hàng, 0 = flat
        self.lastest_pos = 0

        # Config logger
        self.exchange = "binance"
        self.class_name = self.__class__.__name__
        strategy_log_name = f'{self.symbol}_{self.exchange}_{self.class_name}'
        self.logger_strategy = setup_logger_global(
            strategy_log_name, strategy_log_name + '.log'
        )
        self.stop_flag = False

        try:
            account_info = {
                "api_key": api_key,
                "secret_key": secret_key,
                "passphrase": passphrase,
            }

            self.client = get_client_exchange(
                exchange_name="binance",
                acc_info=account_info,
                symbol=self.symbol,
                quote=self.quote,
                session_key=session_key,
            )

            self.logger_strategy.info(
                f"client initialized successfully for {self.symbol}/{self.quote}"
            )
        except Exception as e:
            self.logger_strategy.error(f"❌ Failed to initialize client: {e}")
            raise

    def get_current_price(self):
        """Get current price of the symbol."""
        try:
            price_data = self.client.get_price()
            if price_data and 'price' in price_data:
                return float(price_data['price'])
            else:
                self.logger_strategy.error("Invalid price response")
                return None
        except Exception as e:
            self.logger_strategy.error(f"Error getting price: {e}")
            update_key_and_insert_error_log(
                self.run_key, self.symbol, get_line_number(),
                self.exchange.upper(), "strategy.py",
                f"Error getting price: {e}"
            )
            return None

    def check_account_balance(self, is_base=False):
        """Check available balance of base (ETH) or quote (USDT)."""
        currency = self.symbol if is_base else self.quote
        try:
            balance_data = self.client.get_account_assets(currency)
            if balance_data and 'data' in balance_data:
                available = float(balance_data['data'].get('available', 0))
                return available
            return 0
        except Exception as e:
            self.logger_strategy.error(f"Error checking {currency} balance: {e}")
            return 0

    def get_candles(self):
        """Fetch OHLCV candles and return as DataFrame."""
        candles_response = self.client.get_candles(
            base=self.symbol,
            quote=self.quote,
            interval=self.interval,
            limit=self.limit
        )

        candles = candles_response.get("candle", [])
        self.logger_strategy.info(f"Raw response: {candles_response}")
        self.logger_strategy.info(
            f"Fetched {len(candles)} candles for {self.symbol}/{self.quote} {self.interval} interval"
        )

        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'close_time', 'quote_volume', 'trades',
                     'taker_base', 'taker_quote', 'ignore']
        )

        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    @staticmethod
    def _compute_cci(df: pd.DataFrame, period: int) -> pd.Series:
        """
        CCI = (TP - SMA(TP)) / (0.015 * MeanDeviation(TP))
        TP = (H + L + C) / 3
        MeanDeviation = mean(|TP_i - SMA(TP)|) over window
        """
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        sma_tp = tp.rolling(period).mean()

        # mean deviation (MAD) trên rolling window
        def _mad(x):
            m = x.mean()
            return np.mean(np.abs(x - m))

        mad = tp.rolling(period).apply(_mad, raw=True)

        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    def get_signal(self):
        """
        Return signal based on CCI cross:
        +1 when CCI crosses above +upper
        -1 when CCI crosses below -lower
        0 otherwise
        """
        df = self.get_candles()
        if df.empty or len(df) < max(2, self.cci_period + 2):
            return 0

        cci = self._compute_cci(df, self.cci_period).dropna()
        if len(cci) < 2:
            return 0

        cci_prev, cci_now = cci.iloc[-2], cci.iloc[-1]

        # Cross up through +upper → BUY
        if (cci_prev <= self.cci_upper) and (cci_now > self.cci_upper):
            return 1

        # Cross down through -lower → SELL
        if (cci_prev >= self.cci_lower) and (cci_now < self.cci_lower):
            return -1

        return 0

    def get_quantity(self, price):
        """Calculate quantity based on trade value and current price."""
        return self.trade_value / price if price else 0

    def place_order(self, side, quantity, price=None, order_type="MARKET"):
        """Place order with balance check."""
        try:
            if side == "BUY":
                balance = self.check_account_balance(is_base=False)
                if balance < self.trade_value:
                    self.logger_strategy.warning(f"Insufficient {self.quote} balance")
                    return False
            elif side == "SELL":
                balance = self.check_account_balance(is_base=True)
                if price is None:
                    price = self.get_current_price()
                if not price or (balance * price < self.trade_value):
                    self.logger_strategy.warning(f"Insufficient {self.symbol} balance")
                    return False

            order_result = self.client.place_order(
                side_order=side,
                quantity=quantity,
                order_type=order_type,
                force='GTC'
            )

            if order_result and order_result.get('code') == 0:
                order_id = order_result.get('data', {}).get('orderId', 'N/A')
                self.lastest_pos = 1 if side == "BUY" else -1
                self.logger_strategy.info(
                    f"✅ {side} order placed successfully, ID={order_id}, Qty={quantity:.6f}"
                )
                return True
            else:
                self.logger_strategy.error(f"❌ Failed to place order: {order_result}")
                return False

        except Exception as e:
            self.logger_strategy.error(f"Error placing order: {e}")
            return False

    def logic(self):
        """Main logic of the strategy using CCI crosses."""
        signal = self.get_signal()
        self.logger_strategy.info(f"Signal: {signal}")

        # Không làm gì nếu không có tín hiệu hoặc trùng hướng với vị thế hiện tại
        if signal == 0 or signal == self.lastest_pos:
            return False

        price = self.get_current_price()
        if not price:
            return False
        quantity = self.get_quantity(price)

        if signal == 1:
            return self.place_order("BUY", quantity, price)
        elif signal == -1:
            return self.place_order("SELL", quantity, price)
        return False

    def run_strategy(self):
        self.logger_strategy.info("🚀 Starting CCI Strategy...")
        try:
            while not self.stop_flag:
                executed = self.logic()
                if executed:
                    self.logger_strategy.info("Trade executed successfully")
                else:
                    self.logger_strategy.info("No trade executed this cycle")
                time.sleep(self.sleep)
        except Exception as e:
            self.logger_strategy.error(f"Strategy error: {e} _ {e.__traceback__.tb_lineno}")
            return False


def main():
    """
    Main function to run the strategy
    """
    params = get_constants()
    SESSION_ID     = params.get("SESSION_ID", "")
    EXCHANGE       = params.get("EXCHANGE", "")
    API_KEY        = params.get("API_KEY", "")
    SECRET_KEY     = params.get("SECRET_KEY", "")
    PASSPHRASE     = params.get("PASSPHRASE", "")
    STRATEGY_NAME  = params.get("STRATEGY_NAME", "")
    PAPER_MODE     = params.get("PAPER_MODE", True)     
    
    if not API_KEY or not SECRET_KEY:
        logger_access.info("❌ API credentials are required")
        return
    
    if not SESSION_ID:
        logger_access.info("❌ Session key is required")
        return
    
    logger_access.info("✅ Environment variables loaded successfully")
    logger_access.info(f"🔑 Session Key: {SESSION_ID}")
    logger_access.info("BTC Test Strategy starting...")

    try:
        # Initialize strategy
        strategy = SimpleStrategy(
            api_key=API_KEY,
            secret_key=SECRET_KEY,
            passphrase=PASSPHRASE,
            session_key=SESSION_ID,
        )
        
        logger_access.info("Test Strategy initialized successfully")

        # Run the strategy loop (internal while True)
        strategy.run_strategy()

    except KeyboardInterrupt:
        logger_access.info("\n🛑 Strategy stopped by user")
        
    except Exception as e:
        logger_access.error(f"❌ Fatal error: {e}")
        update_key_and_insert_error_log(
            generate_random_string(),
            "ETH",
            get_line_number(),
            "POLONIEX",
            "test-strategy.py",
            f"Fatal error: {e}"
        )

if __name__ == "__main__":
    main()
