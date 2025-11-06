"""
デイトレード戦略のデバッグ版
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
from src.backtest.engine import BacktestEngine, OrderSide
from src.strategies.base_strategy import BaseStrategy


def analyze_data():
    """データの詳細分析"""
    print("=== データ分析 ===")
    
    # Bloomberg APIから直近データ取得
    SYMBOL = "7203 JT Equity"
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        # 直近7日間のデータ
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 1分足データ取得
        bar_data = fetcher.fetch_intraday_bars(SYMBOL, start_date, end_date, interval=1)
        
        if bar_data.empty:
            print("[ERROR] データが取得できませんでした")
            return
        
        print(f"[INFO] 取得データ: {len(bar_data)}本")
        
        # 時間帯別分析
        bar_data['hour'] = pd.to_datetime(bar_data['time']).dt.hour
        bar_data['date'] = pd.to_datetime(bar_data['time']).dt.date
        
        print("\n【日別データ数】")
        daily_counts = bar_data.groupby('date').size()
        for date, count in daily_counts.items():
            print(f"  {date}: {count}本")
        
        print("\n【時間帯別データ数】")
        hourly_counts = bar_data.groupby('hour').size()
        for hour, count in hourly_counts.items():
            print(f"  {hour}時: {count}本")
        
        print("\n【価格統計】")
        print(f"  最高値: {bar_data['high'].max():.1f}円")
        print(f"  最安値: {bar_data['low'].min():.1f}円")
        print(f"  平均値: {bar_data['close'].mean():.1f}円")
        print(f"  標準偏差: {bar_data['close'].std():.1f}円")
        
        # 価格変動率
        bar_data['return'] = bar_data['close'].pct_change()
        print(f"\n【価格変動率統計】")
        print(f"  平均: {bar_data['return'].mean():.4%}")
        print(f"  標準偏差: {bar_data['return'].std():.4%}")
        print(f"  最大上昇率: {bar_data['return'].max():.4%}")
        print(f"  最大下落率: {bar_data['return'].min():.4%}")
        
        # 移動平均クロスの回数を確認
        bar_data['MA5'] = bar_data['close'].rolling(5).mean()
        bar_data['MA15'] = bar_data['close'].rolling(15).mean()
        bar_data['MA10'] = bar_data['close'].rolling(10).mean()
        bar_data['MA30'] = bar_data['close'].rolling(30).mean()
        
        # クロスの検出
        bar_data['cross_5_15'] = np.where(
            (bar_data['MA5'] > bar_data['MA15']) & 
            (bar_data['MA5'].shift(1) <= bar_data['MA15'].shift(1)), 1, 0)
        bar_data['cross_10_30'] = np.where(
            (bar_data['MA10'] > bar_data['MA30']) & 
            (bar_data['MA10'].shift(1) <= bar_data['MA30'].shift(1)), 1, 0)
        
        print(f"\n【移動平均クロス回数】")
        print(f"  MA5×MA15: {bar_data['cross_5_15'].sum()}回")
        print(f"  MA10×MA30: {bar_data['cross_10_30'].sum()}回")
        
        # クロスが発生した時刻
        crosses_5_15 = bar_data[bar_data['cross_5_15'] == 1][['time', 'close', 'MA5', 'MA15']]
        if not crosses_5_15.empty:
            print(f"\n【MA5×MA15クロス詳細（最初の5件）】")
            for _, row in crosses_5_15.head(5).iterrows():
                print(f"  {row['time']}: 価格={row['close']:.1f}, MA5={row['MA5']:.1f}, MA15={row['MA15']:.1f}")
        
        # モメンタム分析
        for lookback in [10, 20]:
            bar_data[f'momentum_{lookback}'] = bar_data['close'].pct_change(lookback)
            threshold_001 = (bar_data[f'momentum_{lookback}'] > 0.001).sum()
            threshold_0005 = (bar_data[f'momentum_{lookback}'] > 0.0005).sum()
            print(f"\n【モメンタム（{lookback}期間）】")
            print(f"  0.1%超: {threshold_001}回")
            print(f"  0.05%超: {threshold_0005}回")
        
        # 簡易バックテスト
        print("\n【簡易シミュレーション】")
        
        # MA5×MA15戦略
        position = False
        entry_price = 0
        trades = []
        
        for i in range(16, len(bar_data)):
            row = bar_data.iloc[i]
            prev_row = bar_data.iloc[i-1]
            
            # 取引時間内のみ
            if row['hour'] < 9 or row['hour'] >= 15:
                continue
            
            # エントリー
            if not position and row['MA5'] > row['MA15'] and prev_row['MA5'] <= prev_row['MA15']:
                position = True
                entry_price = row['close']
                trades.append({
                    'entry_time': row['time'],
                    'entry_price': entry_price,
                    'type': 'MA_Cross'
                })
            
            # エグジット
            elif position and (row['MA5'] < row['MA15'] or row['hour'] >= 14 and row.name == bar_data.index[-1]):
                exit_price = row['close']
                profit = (exit_price - entry_price) / entry_price
                trades[-1].update({
                    'exit_time': row['time'],
                    'exit_price': exit_price,
                    'profit': profit
                })
                position = False
        
        print(f"  取引回数: {len([t for t in trades if 'exit_time' in t])}回")
        if trades:
            profits = [t['profit'] for t in trades if 'profit' in t]
            if profits:
                print(f"  平均利益率: {np.mean(profits):.3%}")
                print(f"  勝率: {len([p for p in profits if p > 0]) / len(profits) * 100:.1f}%")
                
                print("\n【取引詳細（最初の3件）】")
                for t in trades[:3]:
                    if 'exit_time' in t:
                        print(f"  {t['entry_time'].strftime('%m/%d %H:%M')} → {t['exit_time'].strftime('%H:%M')}: {t['profit']:.3%}")
        
        return bar_data
        
    finally:
        fetcher.disconnect()


if __name__ == "__main__":
    bar_data = analyze_data()
    
    if bar_data is not None:
        # CSVに保存
        csv_path = f"reports/debug_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs("reports", exist_ok=True)
        bar_data.to_csv(csv_path, index=False)
        print(f"\n[INFO] デバッグデータを保存: {csv_path}")