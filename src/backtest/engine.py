import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from datetime import datetime
from loguru import logger
from enum import Enum


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """注文クラス"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0


@dataclass
class Trade:
    """約定クラス"""
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    slippage: float


@dataclass
class Position:
    """ポジションクラス"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


class BacktestEngine:
    """イベント駆動型バックテストエンジン"""
    
    def __init__(self,
                 initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        self.current_time = None
        self.strategy = None
        
    def set_strategy(self, strategy):
        """戦略をセット"""
        self.strategy = strategy
        self.strategy.set_engine(self)
        
    def run(self, tick_data: pd.DataFrame):
        """バックテスト実行"""
        logger.info(f"Starting backtest with {len(tick_data)} ticks")
        
        # ティックデータをイベントとして処理
        for idx, tick in tick_data.iterrows():
            self.current_time = tick['time']
            
            # ポジションの時価評価更新
            self._update_positions(tick)
            
            # ペンディング注文の処理
            self._process_orders(tick)
            
            # 戦略にティックを渡す
            if self.strategy:
                self.strategy.on_tick(tick)
            
            # エクイティカーブを記録
            self._record_equity()
        
        logger.info("Backtest completed")
        
    def place_order(self, order: Order) -> str:
        """注文を発注"""
        order.timestamp = self.current_time
        self.orders.append(order)
        logger.debug(f"Order placed: {order}")
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセル"""
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                logger.debug(f"Order cancelled: {order_id}")
                return True
        return False
    
    def _process_orders(self, tick: pd.Series):
        """注文処理"""
        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue
            
            if order.symbol != tick.get('symbol', ''):
                continue
            
            # 約定判定
            filled = False
            fill_price = 0.0
            
            if tick['type'] == 'TRADE':
                if order.order_type == OrderType.MARKET:
                    filled = True
                    fill_price = tick['value']
                    
                elif order.order_type == OrderType.LIMIT:
                    if (order.side == OrderSide.BUY and tick['value'] <= order.price) or \
                       (order.side == OrderSide.SELL and tick['value'] >= order.price):
                        filled = True
                        fill_price = order.price
            
            if filled:
                self._execute_order(order, fill_price, tick)
    
    def _execute_order(self, order: Order, fill_price: float, tick: pd.Series):
        """注文を約定"""
        # スリッページを適用
        if order.side == OrderSide.BUY:
            execution_price = fill_price * (1 + self.slippage_rate)
        else:
            execution_price = fill_price * (1 - self.slippage_rate)
        
        # 手数料計算
        commission = execution_price * order.quantity * self.commission_rate
        
        # 約定を記録
        trade = Trade(
            trade_id=f"T{len(self.trades)}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            timestamp=tick['time'],
            commission=commission,
            slippage=execution_price - fill_price
        )
        
        self.trades.append(trade)
        
        # ポジション更新
        self._update_position(trade)
        
        # 資金更新
        if order.side == OrderSide.BUY:
            self.capital -= (execution_price * order.quantity + commission)
        else:
            self.capital += (execution_price * order.quantity - commission)
        
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission
        
        logger.debug(f"Order filled: {order.order_id} at {execution_price}")
        
        # 戦略に通知
        if self.strategy:
            self.strategy.on_order_filled(order, trade)
    
    def _update_position(self, trade: Trade):
        """ポジションを更新"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_price=0,
                current_price=trade.price,
                unrealized_pnl=0,
                realized_pnl=0
            )
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # 買い増し
            total_value = position.quantity * position.avg_price + \
                         trade.quantity * trade.price
            position.quantity += trade.quantity
            position.avg_price = total_value / position.quantity if position.quantity > 0 else 0
        else:
            # 売却
            if position.quantity >= trade.quantity:
                # 実現損益を計算
                realized = (trade.price - position.avg_price) * trade.quantity
                position.realized_pnl += realized
                position.quantity -= trade.quantity
            else:
                logger.warning(f"Trying to sell more than owned: {symbol}")
    
    def _update_positions(self, tick: pd.Series):
        """ポジションの時価評価を更新"""
        if tick['type'] != 'TRADE':
            return
        
        symbol = tick.get('symbol', '')
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = tick['value']
            position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
    
    def _record_equity(self):
        """エクイティカーブを記録"""
        total_value = self.capital
        
        # ポジションの時価を加算
        for position in self.positions.values():
            total_value += position.quantity * position.current_price
        
        self.equity_curve.append({
            'timestamp': self.current_time,
            'equity': total_value,
            'cash': self.capital,
            'positions_value': total_value - self.capital
        })
    
    def get_performance_metrics(self) -> Dict:
        """パフォーマンス指標を計算"""
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # リターン計算
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 日次リターン
        daily_returns = equity_df['equity'].pct_change().dropna()
        
        # シャープレシオ（年率換算）
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # 最大ドローダウン
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 勝率
        winning_trades = [t for t in self.trades if t.side == OrderSide.SELL and 
                         self.positions.get(t.symbol, Position(t.symbol, 0, 0, 0, 0, 0)).realized_pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital
        }