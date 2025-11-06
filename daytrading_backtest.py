"""
日中完結型のデイトレード戦略バックテスト
より長期間のデータで検証
"""
import os
import sys
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import json

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
from src.backtest.engine import BacktestEngine, OrderSide
from src.strategies.base_strategy import BaseStrategy


class DayTradingMAStrategy(BaseStrategy):
    """日中完結型の移動平均クロス戦略"""
    
    def __init__(self, fast_period=10, slow_period=30, 
                 open_time="09:00", close_time="15:00", 
                 force_close_time="14:55"):
        super().__init__(name=f"DayTradeMA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.open_time = open_time
        self.close_time = close_time
        self.force_close_time = force_close_time
        
        self.daily_data = {}  # 日付ごとのデータを管理
        self.current_date = None
        self.position_open = False
        self.daily_trades = 0
        self.max_trades_per_day = 10  # 1日の最大取引回数
        
    def on_tick(self, tick):
        """ティックごとの処理"""
        tick_time = tick['time']
        tick_date = tick_time.date()
        time_only = tick_time.time()
        
        # 新しい日の開始
        if tick_date != self.current_date:
            self.current_date = tick_date
            self.daily_data[tick_date] = []
            self.daily_trades = 0
            self.position_open = False
        
        # 取引時間外はスキップ
        if time_only < pd.to_datetime(self.open_time).time():
            return
        if time_only > pd.to_datetime(self.close_time).time():
            return
        
        # データを追加
        self.daily_data[tick_date].append({
            'time': tick_time,
            'close': tick.get('close', tick.get('value', 0))
        })
        
        # 必要なデータが揃っていない場合はスキップ
        if len(self.daily_data[tick_date]) < self.slow_period:
            return
        
        # 移動平均を計算（当日のデータのみ使用）
        closes = [bar['close'] for bar in self.daily_data[tick_date][-self.slow_period:]]
        fast_ma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_ma = sum(closes) / self.slow_period
        
        # 前回の移動平均
        if len(self.daily_data[tick_date]) > self.slow_period:
            prev_closes = [bar['close'] for bar in self.daily_data[tick_date][-self.slow_period-1:-1]]
            prev_fast_ma = sum(prev_closes[-self.fast_period:]) / self.fast_period
            prev_slow_ma = sum(prev_closes) / self.slow_period
        else:
            return  # 初回はスキップ
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # 強制クローズ時間のチェック（14:55以降）
        if time_only >= pd.to_datetime(self.force_close_time).time():
            if self.position_open and position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.position_open = False
                self.log_signal("force_close", True)
                print(f"[FORCE CLOSE] {tick_date} {time_only} - 日中クローズ")
            return
        
        # 1日の取引回数制限チェック
        if self.daily_trades >= self.max_trades_per_day:
            return
        
        # エントリー条件（ゴールデンクロス）
        if (prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and 
            not self.position_open and time_only <= pd.to_datetime("14:30").time()):
            cash = self.get_cash_balance()
            price = tick.get('close', tick.get('value', 0))
            if price > 0:
                quantity = int((cash * 0.95) / price / 100) * 100
                if quantity > 0:
                    self.place_market_order(symbol, OrderSide.BUY, quantity)
                    self.position_open = True
                    self.daily_trades += 1
                    self.log_signal("golden_cross", True)
        
        # エグジット条件（デッドクロス）
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and self.position_open:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.position_open = False
                self.daily_trades += 1
                self.log_signal("dead_cross", True)


class DayTradingBreakoutStrategy(BaseStrategy):
    """日中ブレイクアウト戦略"""
    
    def __init__(self, lookback_minutes=30, breakout_threshold=0.002,
                 open_time="09:00", close_time="15:00", 
                 force_close_time="14:55"):
        super().__init__(name=f"DayTradeBreakout_{lookback_minutes}")
        self.lookback_minutes = lookback_minutes
        self.breakout_threshold = breakout_threshold
        self.open_time = open_time
        self.close_time = close_time
        self.force_close_time = force_close_time
        
        self.daily_data = {}
        self.current_date = None
        self.position_open = False
        self.daily_trades = 0
        self.max_trades_per_day = 5
        self.entry_price = None
        
    def on_tick(self, tick):
        """ティックごとの処理"""
        tick_time = tick['time']
        tick_date = tick_time.date()
        time_only = tick_time.time()
        
        # 新しい日の開始
        if tick_date != self.current_date:
            self.current_date = tick_date
            self.daily_data[tick_date] = []
            self.daily_trades = 0
            self.position_open = False
            self.entry_price = None
        
        # 取引時間外はスキップ
        if time_only < pd.to_datetime(self.open_time).time():
            return
        if time_only > pd.to_datetime(self.close_time).time():
            return
        
        # データを追加
        price = tick.get('close', tick.get('value', 0))
        self.daily_data[tick_date].append({
            'time': tick_time,
            'price': price,
            'volume': tick.get('size', 0)
        })
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # 強制クローズ
        if time_only >= pd.to_datetime(self.force_close_time).time():
            if self.position_open and position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.position_open = False
                print(f"[FORCE CLOSE] {tick_date} {time_only}")
            return
        
        # データが十分でない場合はスキップ
        if len(self.daily_data[tick_date]) < self.lookback_minutes:
            return
        
        # 直近のデータから高値・安値を計算
        recent_prices = [d['price'] for d in self.daily_data[tick_date][-self.lookback_minutes:]]
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        
        # ポジションなしの場合のエントリー判定
        if not self.position_open and self.daily_trades < self.max_trades_per_day:
            # 上方ブレイクアウト
            if price > recent_high * (1 + self.breakout_threshold):
                cash = self.get_cash_balance()
                quantity = int((cash * 0.95) / price / 100) * 100
                if quantity > 0:
                    self.place_market_order(symbol, OrderSide.BUY, quantity)
                    self.position_open = True
                    self.entry_price = price
                    self.daily_trades += 1
                    print(f"[BREAKOUT BUY] {tick_date} {time_only} @ {price:.1f}")
        
        # ポジションありの場合のエグジット判定
        elif self.position_open and position and position.quantity > 0:
            # 利確（エントリー価格の0.5%上昇）
            if price > self.entry_price * 1.005:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.position_open = False
                print(f"[TAKE PROFIT] {tick_date} {time_only} @ {price:.1f}")
            
            # 損切り（エントリー価格の0.3%下落）
            elif price < self.entry_price * 0.997:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.position_open = False
                print(f"[STOP LOSS] {tick_date} {time_only} @ {price:.1f}")


def fetch_long_term_data(symbol, days=30):
    """長期間のデータを取得"""
    print(f"[INFO] {days}日分のデータを取得中...")
    
    fetcher = TickDataFetcher()
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return None
    
    try:
        all_data = []
        end_date = datetime.now()
        
        # 日次でデータを取得（APIの制限を考慮）
        for i in range(days):
            current_date = end_date - timedelta(days=i)
            
            # 土日はスキップ
            if current_date.weekday() >= 5:  # 5=土曜, 6=日曜
                continue
            
            start_time = current_date.replace(hour=9, minute=0, second=0)
            end_time = current_date.replace(hour=15, minute=0, second=0)
            
            print(f"  {current_date.strftime('%Y-%m-%d')}のデータ取得中...", end="")
            
            try:
                # 5分足データを取得（ティックデータは大量すぎるため）
                bar_data = fetcher.fetch_intraday_bars(
                    symbol, start_time, end_time, interval=5
                )
                
                if not bar_data.empty:
                    all_data.append(bar_data)
                    print(f" {len(bar_data)}本")
                else:
                    print(" データなし")
                    
            except Exception as e:
                print(f" エラー: {e}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"[INFO] 合計 {len(combined_data)}本のデータを取得")
            return combined_data
        else:
            return None
            
    finally:
        fetcher.disconnect()


def run_daytrading_backtest():
    """デイトレード戦略のバックテスト実行"""
    print("=== 日本株デイトレード戦略バックテスト ===")
    print("期間: 過去30日間")
    print("制約: 日中完結（14:55に強制クローズ）\n")
    
    # データ取得
    SYMBOL = "7203 JT Equity"
    bar_data = fetch_long_term_data(SYMBOL, days=30)
    
    if bar_data is None or bar_data.empty:
        print("[ERROR] データが取得できませんでした")
        return
    
    # 戦略リスト
    strategies = [
        # 移動平均クロス戦略（異なるパラメータ）
        DayTradingMAStrategy(fast_period=5, slow_period=20),
        DayTradingMAStrategy(fast_period=10, slow_period=30),
        DayTradingMAStrategy(fast_period=15, slow_period=45),
        
        # ブレイクアウト戦略
        DayTradingBreakoutStrategy(lookback_minutes=30, breakout_threshold=0.002),
        DayTradingBreakoutStrategy(lookback_minutes=60, breakout_threshold=0.003),
    ]
    
    results = []
    
    for strategy in strategies:
        print(f"\n[INFO] テスト中: {strategy.name}")
        
        # ティック形式に変換
        tick_data = []
        for _, bar in bar_data.iterrows():
            tick_data.append({
                'time': bar['time'],
                'type': 'TRADE',
                'symbol': SYMBOL,
                'value': bar['close'],
                'close': bar['close'],
                'size': bar['volume'] // 4
            })
        
        tick_df = pd.DataFrame(tick_data)
        
        # バックテスト実行
        engine = BacktestEngine(
            initial_capital=10_000_000,
            commission_rate=0.0003,
            slippage_rate=0.0001
        )
        
        engine.set_strategy(strategy)
        engine.run(tick_df)
        
        # 結果を保存
        metrics = engine.get_performance_metrics()
        
        # 日次リターンを計算
        equity_df = pd.DataFrame(engine.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
            daily_equity = equity_df.resample('D').last()
            daily_returns = daily_equity['equity'].pct_change().dropna()
            
            # 追加メトリクス
            winning_days = (daily_returns > 0).sum()
            total_days = len(daily_returns)
            daily_win_rate = winning_days / total_days if total_days > 0 else 0
            
            metrics['daily_win_rate'] = daily_win_rate
            metrics['total_days'] = total_days
            metrics['avg_daily_return'] = daily_returns.mean()
            metrics['daily_volatility'] = daily_returns.std()
            
        results.append({
            'strategy': strategy.name,
            **metrics
        })
        
        print(f"  総リターン: {metrics.get('total_return', 0):.2%}")
        print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  日次勝率: {metrics.get('daily_win_rate', 0):.1%}")
        print(f"  総取引数: {metrics.get('total_trades', 0)}")
    
    # 結果をDataFrameにまとめる
    results_df = pd.DataFrame(results)
    
    # HTMLレポート生成
    create_daytrading_report(results_df, bar_data)
    
    # CSVで保存
    csv_path = f"reports/daytrading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 結果を保存: {csv_path}")


def create_daytrading_report(results_df, bar_data):
    """デイトレード結果のHTMLレポート生成"""
    
    # 日付範囲
    start_date = bar_data['time'].min().strftime('%Y年%m月%d日')
    end_date = bar_data['time'].max().strftime('%Y年%m月%d日')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>デイトレード戦略バックテスト結果</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .strategy-card {{
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }}
        .strategy-card:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }}
        .strategy-name {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .metric {{
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            background-color: #f8f9fa;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .neutral {{ color: #3498db; }}
        .best-strategy {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border: 2px solid #667eea;
        }}
        .summary {{
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 30px;
        }}
        h2 {{
            color: #333;
            margin-top: 30px;
            margin-bottom: 20px;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>トヨタ自動車 (7203) デイトレード戦略バックテスト</h1>
        <p>期間: {start_date} ～ {end_date}</p>
        <p>制約: 全ポジション日中完結（14:55強制クローズ）</p>
    </div>
    
    <h2>戦略別パフォーマンス</h2>
"""
    
    # 最良の戦略を特定
    best_idx = results_df['sharpe_ratio'].idxmax() if not results_df.empty else None
    
    # 各戦略の結果を表示
    for idx, row in results_df.iterrows():
        is_best = idx == best_idx
        card_class = "strategy-card best-strategy" if is_best else "strategy-card"
        
        # 色分け
        return_class = "positive" if row['total_return'] > 0 else "negative"
        sharpe_class = "positive" if row['sharpe_ratio'] > 0 else "negative"
        
        html_content += f"""
    <div class="{card_class}">
        <div class="strategy-name">
            {row['strategy']}
            {' ⭐ ベスト戦略' if is_best else ''}
        </div>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-label">総リターン</div>
                <div class="metric-value {return_class}">{row['total_return']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">シャープレシオ</div>
                <div class="metric-value {sharpe_class}">{row['sharpe_ratio']:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">最大DD</div>
                <div class="metric-value negative">{row['max_drawdown']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">日次勝率</div>
                <div class="metric-value neutral">{row.get('daily_win_rate', 0):.1%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">総取引数</div>
                <div class="metric-value neutral">{row['total_trades']:.0f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">1日平均取引</div>
                <div class="metric-value neutral">{row['total_trades']/row.get('total_days', 1):.1f}</div>
            </div>
        </div>
    </div>
"""
    
    # サマリー統計
    avg_return = results_df['total_return'].mean()
    positive_strategies = (results_df['total_return'] > 0).sum()
    
    html_content += f"""
    <div class="summary">
        <h2>総合分析</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-label">戦略数</div>
                <div class="metric-value">{len(results_df)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">プラス戦略</div>
                <div class="metric-value">{positive_strategies}/{len(results_df)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">平均リターン</div>
                <div class="metric-value {('positive' if avg_return > 0 else 'negative')}">{avg_return:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">最高リターン</div>
                <div class="metric-value positive">{results_df['total_return'].max():.2%}</div>
            </div>
        </div>
    </div>
    
    <div class="warning">
        <strong>注意事項:</strong>
        <ul>
            <li>このバックテストは過去のデータに基づくものであり、将来の収益を保証するものではありません</li>
            <li>実際の取引では、スリッページや流動性の問題がより大きく影響する可能性があります</li>
            <li>デイトレードは高頻度取引となるため、手数料の影響が大きくなります</li>
            <li>市場環境の変化により、戦略の有効性は変動します</li>
        </ul>
    </div>
</body>
</html>
"""
    
    # ファイル保存
    filename = f"reports/daytrading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[INFO] レポートを生成: {filename}")
    
    # ブラウザで開く
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(filename)}')


if __name__ == "__main__":
    run_daytrading_backtest()