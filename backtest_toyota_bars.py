"""
取得したトヨタの分足データを使ってバックテストを実行
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
from src.backtest.engine import BacktestEngine, Order, OrderType, OrderSide
from src.strategies.base_strategy import BaseStrategy

# ログ設定
class SimpleLogger:
    @staticmethod
    def info(msg):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    @staticmethod
    def error(msg):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {msg}")

logger = SimpleLogger()


class SimpleMAStrategy(BaseStrategy):
    """シンプルな移動平均クロス戦略"""
    
    def __init__(self, fast_period=10, slow_period=30):
        super().__init__(name="SimpleMAStrategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.bar_data = []
        self.position_open = False
        
    def on_tick(self, tick):
        """分足データをティックとして処理"""
        # バーデータとして保存
        self.bar_data.append({
            'time': tick['time'],
            'close': tick.get('close', tick.get('value', 0))
        })
        
        # 必要なバー数が揃っていない場合はスキップ
        if len(self.bar_data) < self.slow_period:
            return
        
        # 移動平均を計算
        closes = [bar['close'] for bar in self.bar_data[-self.slow_period:]]
        fast_ma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_ma = sum(closes) / self.slow_period
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # ゴールデンクロス（買いシグナル）
        if fast_ma > slow_ma and not self.position_open:
            cash = self.get_cash_balance()
            price = tick.get('close', tick.get('value', 0))
            if price > 0:
                quantity = int((cash * 0.95) / price / 100) * 100  # 100株単位
                if quantity > 0:
                    self.place_market_order(symbol, OrderSide.BUY, quantity)
                    self.position_open = True
                    logger.info(f"Buy signal: Fast MA ({fast_ma:.2f}) > Slow MA ({slow_ma:.2f})")
        
        # デッドクロス（売りシグナル）
        elif fast_ma < slow_ma and self.position_open:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.position_open = False
                logger.info(f"Sell signal: Fast MA ({fast_ma:.2f}) < Slow MA ({slow_ma:.2f})")


def convert_bars_to_ticks(bar_data):
    """分足データをティック形式に変換"""
    ticks = []
    for _, bar in bar_data.iterrows():
        # 各バーを4つのティックに変換（OHLC）
        base_tick = {
            'time': bar['time'],
            'type': 'TRADE',
            'symbol': '7203 JT Equity',
            'exchange': 'TSE',
            'condition': ''
        }
        
        # Open
        ticks.append({**base_tick, 'value': bar['open'], 'size': bar['volume'] // 4})
        # High
        ticks.append({**base_tick, 'value': bar['high'], 'size': bar['volume'] // 4})
        # Low
        ticks.append({**base_tick, 'value': bar['low'], 'size': bar['volume'] // 4})
        # Close
        ticks.append({**base_tick, 'value': bar['close'], 'size': bar['volume'] // 4})
    
    return pd.DataFrame(ticks)


def run_backtest_with_bars():
    """分足データでバックテスト実行"""
    logger.info("=== Toyota Bar Data Backtest ===")
    
    # Bloomberg APIから分足データ取得
    SYMBOL = "7203 JT Equity"
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=7)
    
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        logger.error("Failed to connect to Bloomberg API")
        return
    
    try:
        logger.info("Fetching bar data...")
        bar_data = fetcher.fetch_intraday_bars(SYMBOL, START_DATE, END_DATE, interval=1)
        
        if bar_data.empty:
            logger.error("No data fetched")
            return
        
        logger.info(f"Fetched: {len(bar_data)} bars")
        
        # 分足データをティック形式に変換
        tick_data = convert_bars_to_ticks(bar_data)
        logger.info(f"Converted to {len(tick_data)} tick events")
        
        # バックテスト実行
        logger.info("\n=== Running Backtest ===")
        
        # 移動平均クロス戦略
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30)
        
        engine = BacktestEngine(
            initial_capital=10_000_000,  # 1000万円
            commission_rate=0.0003,      # 0.03%
            slippage_rate=0.0001         # 0.01%
        )
        
        engine.set_strategy(strategy)
        engine.run(tick_data)
        
        # 結果表示
        metrics = engine.get_performance_metrics()
        
        logger.info("\n=== Backtest Results ===")
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Final Equity: {metrics.get('final_equity', 0):,.0f} yen")
        
        # 取引詳細
        if engine.trades:
            logger.info(f"\n=== Trade Summary ===")
            logger.info(f"Number of trades: {len(engine.trades)}")
            for i, trade in enumerate(engine.trades[:10]):  # 最初の10件
                logger.info(f"Trade {i+1}: {trade.side.value} {trade.quantity} shares "
                          f"@ {trade.price:.1f} yen")
        
        # エクイティカーブ保存
        if engine.equity_curve:
            equity_df = pd.DataFrame(engine.equity_curve)
            csv_path = f"reports/toyota_ma_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs("reports", exist_ok=True)
            equity_df.to_csv(csv_path, index=False)
            logger.info(f"\nEquity curve saved: {csv_path}")
            
            # 最終結果サマリー
            initial = engine.initial_capital
            final = equity_df['equity'].iloc[-1]
            total_return = (final - initial) / initial * 100
            
            logger.info(f"\n=== Final Summary ===")
            logger.info(f"Initial Capital: {initial:,.0f} yen")
            logger.info(f"Final Capital: {final:,.0f} yen")
            logger.info(f"Total Return: {total_return:.2f}%")
            logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()
        logger.info("\n=== Completed ===")


if __name__ == "__main__":
    run_backtest_with_bars()