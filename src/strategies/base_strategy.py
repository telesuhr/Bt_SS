from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger
from ..backtest.engine import Order, OrderType, OrderSide, Trade


class BaseStrategy(ABC):
    """戦略の基底クラス"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.engine = None
        self.indicators = {}
        self.signals = {}
        self.params = {}
        
    def set_engine(self, engine):
        """バックテストエンジンをセット"""
        self.engine = engine
        
    @abstractmethod
    def on_tick(self, tick: pd.Series):
        """ティックデータ受信時の処理"""
        pass
    
    def on_bar(self, bar: pd.Series):
        """バーデータ受信時の処理（オプション）"""
        pass
    
    def on_order_filled(self, order: Order, trade: Trade):
        """注文約定時の処理"""
        logger.info(f"Order filled: {order.order_id} at {trade.price}")
    
    def place_market_order(self, symbol: str, side: OrderSide, quantity: int) -> str:
        """成行注文を発注"""
        order = Order(
            order_id=f"O{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity
        )
        return self.engine.place_order(order)
    
    def place_limit_order(self, symbol: str, side: OrderSide, 
                         quantity: int, price: float) -> str:
        """指値注文を発注"""
        order = Order(
            order_id=f"O{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        return self.engine.place_order(order)
    
    def get_position(self, symbol: str):
        """現在のポジションを取得"""
        return self.engine.positions.get(symbol)
    
    def get_cash_balance(self) -> float:
        """現金残高を取得"""
        return self.engine.capital
    
    def log_signal(self, signal_name: str, value: any):
        """シグナルをログ"""
        self.signals[signal_name] = value
        logger.debug(f"Signal: {signal_name} = {value}")


class TickMomentumStrategy(BaseStrategy):
    """ティックデータのモメンタム戦略サンプル"""
    
    def __init__(self, 
                 lookback_ticks: int = 100,
                 entry_threshold: float = 0.001,
                 exit_threshold: float = -0.0005):
        super().__init__(name="TickMomentumStrategy")
        
        self.lookback_ticks = lookback_ticks
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        self.tick_prices = []
        self.in_position = False
        
    def on_tick(self, tick: pd.Series):
        """ティックごとの処理"""
        if tick['type'] != 'TRADE':
            return
        
        # 価格履歴を更新
        self.tick_prices.append(tick['value'])
        if len(self.tick_prices) > self.lookback_ticks:
            self.tick_prices.pop(0)
        
        if len(self.tick_prices) < self.lookback_ticks:
            return
        
        # モメンタム計算
        momentum = (self.tick_prices[-1] - self.tick_prices[0]) / self.tick_prices[0]
        self.log_signal("momentum", momentum)
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # エントリー判定
        if not self.in_position and momentum > self.entry_threshold:
            # 資金の50%を使用
            cash = self.get_cash_balance()
            quantity = int((cash * 0.5) / tick['value'])
            
            if quantity > 0:
                self.place_market_order(symbol, OrderSide.BUY, quantity)
                self.in_position = True
                logger.info(f"Entry signal: momentum={momentum:.4f}")
        
        # エグジット判定
        elif self.in_position and momentum < self.exit_threshold:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.in_position = False
                logger.info(f"Exit signal: momentum={momentum:.4f}")


class SpreadArbitrageStrategy(BaseStrategy):
    """スプレッドアービトラージ戦略サンプル"""
    
    def __init__(self,
                 spread_threshold: float = 0.0005,
                 position_size: int = 1000):
        super().__init__(name="SpreadArbitrageStrategy")
        
        self.spread_threshold = spread_threshold
        self.position_size = position_size
        
        self.last_bid = None
        self.last_ask = None
        self.mid_price_history = []
        
    def on_tick(self, tick: pd.Series):
        """ティックごとの処理"""
        # BID/ASK更新
        if tick['type'] == 'BID':
            self.last_bid = tick['value']
        elif tick['type'] == 'ASK':
            self.last_ask = tick['value']
        
        # スプレッド計算
        if self.last_bid and self.last_ask:
            spread = self.last_ask - self.last_bid
            mid_price = (self.last_ask + self.last_bid) / 2
            relative_spread = spread / mid_price
            
            self.log_signal("spread", spread)
            self.log_signal("relative_spread", relative_spread)
            
            # 中値の履歴を保持
            self.mid_price_history.append(mid_price)
            if len(self.mid_price_history) > 100:
                self.mid_price_history.pop(0)
            
            # スプレッドが閾値を超えた場合の処理
            if relative_spread > self.spread_threshold and len(self.mid_price_history) >= 50:
                avg_mid = sum(self.mid_price_history[-50:]) / 50
                
                symbol = tick.get('symbol', 'UNKNOWN')
                position = self.get_position(symbol)
                
                # 現在の中値が平均より低い場合は買い
                if mid_price < avg_mid * 0.999 and (not position or position.quantity == 0):
                    self.place_limit_order(symbol, OrderSide.BUY, 
                                         self.position_size, self.last_bid)
                    logger.info(f"Spread arbitrage BUY: spread={relative_spread:.4f}")
                
                # 現在の中値が平均より高い場合は売り
                elif mid_price > avg_mid * 1.001 and position and position.quantity > 0:
                    self.place_limit_order(symbol, OrderSide.SELL,
                                         position.quantity, self.last_ask)
                    logger.info(f"Spread arbitrage SELL: spread={relative_spread:.4f}")


class VWAPStrategy(BaseStrategy):
    """VWAP戦略サンプル"""
    
    def __init__(self,
                 vwap_window: int = 500,
                 deviation_threshold: float = 0.002):
        super().__init__(name="VWAPStrategy")
        
        self.vwap_window = vwap_window
        self.deviation_threshold = deviation_threshold
        
        self.price_volume_data = []
        
    def calculate_vwap(self) -> Optional[float]:
        """VWAPを計算"""
        if not self.price_volume_data:
            return None
        
        total_pv = sum(p * v for p, v in self.price_volume_data)
        total_volume = sum(v for _, v in self.price_volume_data)
        
        return total_pv / total_volume if total_volume > 0 else None
    
    def on_tick(self, tick: pd.Series):
        """ティックごとの処理"""
        if tick['type'] != 'TRADE':
            return
        
        # 価格とボリュームを記録
        self.price_volume_data.append((tick['value'], tick['size']))
        if len(self.price_volume_data) > self.vwap_window:
            self.price_volume_data.pop(0)
        
        # VWAP計算
        vwap = self.calculate_vwap()
        if not vwap:
            return
        
        self.log_signal("vwap", vwap)
        
        # 現在価格とVWAPの乖離
        deviation = (tick['value'] - vwap) / vwap
        self.log_signal("vwap_deviation", deviation)
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # VWAPより大幅に安い場合は買い
        if deviation < -self.deviation_threshold:
            if not position or position.quantity == 0:
                cash = self.get_cash_balance()
                quantity = int((cash * 0.3) / tick['value'])
                
                if quantity > 0:
                    self.place_market_order(symbol, OrderSide.BUY, quantity)
                    logger.info(f"VWAP BUY: price={tick['value']}, vwap={vwap:.2f}, deviation={deviation:.4f}")
        
        # VWAPより大幅に高い場合は売り
        elif deviation > self.deviation_threshold:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                logger.info(f"VWAP SELL: price={tick['value']}, vwap={vwap:.2f}, deviation={deviation:.4f}")