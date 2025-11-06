"""
時刻修正版のデイトレード戦略
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.backtest.engine import BacktestEngine, OrderSide
from src.strategies.base_strategy import BaseStrategy


class SimpleDayTradingStrategy(BaseStrategy):
    """シンプルなデイトレード戦略（時刻修正版）"""
    
    def __init__(self, name="SimpleDayTrading"):
        super().__init__(name=name)
        self.price_history = []
        self.has_position = False
        self.entry_price = None
        self.trades_today = 0
        self.current_date = None
        
    def on_tick(self, tick):
        price = tick.get('close', tick.get('value', 0))
        tick_time = tick['time']
        tick_hour = tick_time.hour
        tick_date = tick_time.date()
        
        # 新しい日の開始
        if tick_date != self.current_date:
            self.current_date = tick_date
            self.price_history = []
            self.trades_today = 0
            # 前日のポジションがあれば強制決済
            if self.has_position:
                self.force_close(tick)
        
        # 価格履歴追加
        self.price_history.append({
            'time': tick_time,
            'price': price,
            'hour': tick_hour
        })
        
        # 直近100本のみ保持
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # データが少ない場合はスキップ
        if len(self.price_history) < 20:
            return
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # 取引時間の判定（0〜6時が実際の9〜15時）
        is_trading_hours = 0 <= tick_hour <= 5  # 0時〜5時59分
        is_close_time = tick_hour == 5  # 5時台（実際の14時台）
        
        # 強制クローズ
        if self.has_position and is_close_time:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.has_position = False
                profit = (price - self.entry_price) / self.entry_price * 100
                print(f"[CLOSE] {tick_time} @ {price:.1f} (損益: {profit:+.2f}%)")
                return
        
        # 取引時間外はスキップ
        if not is_trading_hours:
            return
        
        # 1日3回までの制限
        if self.trades_today >= 3:
            return
        
        # エントリー条件：直近の価格上昇
        recent_prices = [h['price'] for h in self.price_history[-20:]]
        price_change = (recent_prices[-1] / recent_prices[0] - 1)
        
        # 買いエントリー（0.2%以上の上昇）
        if not self.has_position and price_change > 0.002 and not is_close_time:
            cash = self.get_cash_balance()
            quantity = int((cash * 0.95) / price / 100) * 100
            if quantity > 0:
                self.place_market_order(symbol, OrderSide.BUY, quantity)
                self.has_position = True
                self.entry_price = price
                self.trades_today += 1
                print(f"[BUY] {tick_time} @ {price:.1f} (上昇率: {price_change:.3%})")
        
        # 売りエグジット
        elif self.has_position and position:
            profit_rate = (price - self.entry_price) / self.entry_price
            
            # 利確（0.3%以上）または損切り（-0.2%以下）
            if profit_rate >= 0.003 or profit_rate <= -0.002:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.has_position = False
                reason = "利確" if profit_rate > 0 else "損切"
                print(f"[{reason}] {tick_time} @ {price:.1f} (損益: {profit_rate*100:+.2f}%)")
    
    def force_close(self, tick):
        """ポジションの強制クローズ"""
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        if position and position.quantity > 0:
            self.place_market_order(symbol, OrderSide.SELL, position.quantity)
            self.has_position = False
            print(f"[FORCE CLOSE] {tick['time']}")


class VolumeBreakoutStrategy(BaseStrategy):
    """出来高ブレイクアウト戦略"""
    
    def __init__(self, name="VolumeBreakout"):
        super().__init__(name=name)
        self.volume_history = []
        self.price_history = []
        self.has_position = False
        self.entry_price = None
        self.current_date = None
        
    def on_tick(self, tick):
        price = tick.get('close', tick.get('value', 0))
        volume = tick.get('size', 0)
        tick_time = tick['time']
        tick_hour = tick_time.hour
        tick_date = tick_time.date()
        
        # 新しい日
        if tick_date != self.current_date:
            self.current_date = tick_date
            self.volume_history = []
            self.price_history = []
            if self.has_position:
                self.force_close(tick)
        
        # データ追加
        self.volume_history.append(volume)
        self.price_history.append(price)
        
        # 直近50本のみ
        if len(self.volume_history) > 50:
            self.volume_history.pop(0)
            self.price_history.pop(0)
        
        if len(self.volume_history) < 20:
            return
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # 取引時間（0〜5時）
        is_trading_hours = 0 <= tick_hour <= 5
        is_close_time = tick_hour == 5
        
        # 強制クローズ
        if self.has_position and is_close_time:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.has_position = False
                profit = (price - self.entry_price) / self.entry_price * 100
                print(f"[V-CLOSE] {tick_time} @ {price:.1f} (損益: {profit:+.2f}%)")
                return
        
        if not is_trading_hours:
            return
        
        # 出来高の移動平均
        avg_volume = np.mean(self.volume_history[:-1])
        current_volume = self.volume_history[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # 価格の動き
        price_change = (self.price_history[-1] / self.price_history[-5] - 1) if len(self.price_history) >= 5 else 0
        
        # エントリー条件：出来高急増＋価格上昇
        if not self.has_position and volume_ratio > 2.0 and price_change > 0.001 and not is_close_time:
            cash = self.get_cash_balance()
            quantity = int((cash * 0.95) / price / 100) * 100
            if quantity > 0:
                self.place_market_order(symbol, OrderSide.BUY, quantity)
                self.has_position = True
                self.entry_price = price
                print(f"[V-BUY] {tick_time} @ {price:.1f} (出来高比: {volume_ratio:.1f}倍)")
        
        # エグジット
        elif self.has_position and position:
            profit_rate = (price - self.entry_price) / self.entry_price
            
            if profit_rate >= 0.002 or profit_rate <= -0.0015:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.has_position = False
                reason = "利確" if profit_rate > 0 else "損切"
                print(f"[V-{reason}] {tick_time} @ {price:.1f} (損益: {profit_rate*100:+.2f}%)")
    
    def force_close(self, tick):
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        if position and position.quantity > 0:
            self.place_market_order(symbol, OrderSide.SELL, position.quantity)
            self.has_position = False


def run_fixed_daytrading():
    """修正版デイトレードの実行"""
    print("=== デイトレード戦略バックテスト（時刻修正版） ===")
    print("注意: データの時刻が0〜6時で表示されています（実際の9〜15時）")
    
    # データ取得
    SYMBOL = "7203 JT Equity"
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 1分足データ取得
        bar_data = fetcher.fetch_intraday_bars(SYMBOL, start_date, end_date, interval=1)
        
        if bar_data.empty:
            print("[ERROR] データが取得できませんでした")
            return
        
        print(f"\n[INFO] データ取得完了: {len(bar_data)}本")
        print(f"[INFO] 期間: {bar_data['time'].min()} ～ {bar_data['time'].max()}")
        
        # 時間帯確認
        bar_data['hour'] = pd.to_datetime(bar_data['time']).dt.hour
        hour_counts = bar_data.groupby('hour').size()
        print("\n[時間帯別データ数]")
        for hour in sorted(hour_counts.index):
            print(f"  {hour}時: {hour_counts[hour]}本")
        
        # ティック形式に変換
        tick_data = []
        for _, bar in bar_data.iterrows():
            tick_data.append({
                'time': bar['time'],
                'type': 'TRADE',
                'symbol': SYMBOL,
                'value': bar['close'],
                'close': bar['close'],
                'size': bar['volume']
            })
        
        tick_df = pd.DataFrame(tick_data)
        
        # 戦略リスト
        strategies = [
            SimpleDayTradingStrategy(),
            VolumeBreakoutStrategy(),
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\n[テスト中] {strategy.name}")
            print("-" * 50)
            
            # バックテスト実行
            engine = BacktestEngine(
                initial_capital=10_000_000,
                commission_rate=0.0003,
                slippage_rate=0.0001
            )
            
            engine.set_strategy(strategy)
            engine.run(tick_df)
            
            # 結果取得
            metrics = engine.get_performance_metrics()
            
            print(f"\n結果:")
            print(f"  総リターン: {metrics.get('total_return', 0):.2%}")
            print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  最大DD: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  総取引数: {metrics.get('total_trades', 0)}")
            print(f"  最終資産: {metrics.get('final_equity', 0):,.0f}円")
            
            results.append({
                'strategy': strategy.name,
                **metrics
            })
        
        # サマリー表示
        print("\n" + "="*70)
        print("総合結果")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        print(f"\n{'戦略':<20} {'リターン':>10} {'シャープ':>10} {'最大DD':>10} {'取引数':>8}")
        print("-"*60)
        for _, row in results_df.iterrows():
            print(f"{row['strategy']:<20} {row['total_return']:>9.2%} "
                  f"{row['sharpe_ratio']:>10.2f} {row['max_drawdown']:>9.2%} "
                  f"{row['total_trades']:>8.0f}")
        
        # CSV保存
        csv_path = f"reports/daytrading_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs("reports", exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\n[INFO] 結果を保存: {csv_path}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()


if __name__ == "__main__":
    run_fixed_daytrading()