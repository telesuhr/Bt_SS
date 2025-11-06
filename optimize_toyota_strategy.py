"""
トヨタ株の最適な時間足とパラメータの組み合わせを検証
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import product
import json

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
from src.backtest.engine import BacktestEngine, OrderSide
from src.strategies.base_strategy import BaseStrategy


class OptimizedMAStrategy(BaseStrategy):
    """最適化用の移動平均クロス戦略"""
    
    def __init__(self, fast_period=10, slow_period=30, min_hold_bars=1):
        super().__init__(name=f"MA_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_hold_bars = min_hold_bars
        self.bar_data = []
        self.position_open = False
        self.bars_since_entry = 0
        
    def on_tick(self, tick):
        """バーデータをティックとして処理"""
        self.bar_data.append({
            'time': tick['time'],
            'close': tick.get('close', tick.get('value', 0))
        })
        
        if len(self.bar_data) < self.slow_period:
            return
        
        # 移動平均を計算
        closes = [bar['close'] for bar in self.bar_data[-self.slow_period:]]
        fast_ma = sum(closes[-self.fast_period:]) / self.fast_period
        slow_ma = sum(closes) / self.slow_period
        
        # 前回の移動平均
        if len(self.bar_data) > self.slow_period:
            prev_closes = [bar['close'] for bar in self.bar_data[-self.slow_period-1:-1]]
            prev_fast_ma = sum(prev_closes[-self.fast_period:]) / self.fast_period
            prev_slow_ma = sum(prev_closes) / self.slow_period
        else:
            prev_fast_ma = fast_ma
            prev_slow_ma = slow_ma
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # ポジション保有期間をカウント
        if self.position_open:
            self.bars_since_entry += 1
        
        # ゴールデンクロス（買いシグナル）
        if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and not self.position_open:
            cash = self.get_cash_balance()
            price = tick.get('close', tick.get('value', 0))
            if price > 0:
                quantity = int((cash * 0.95) / price / 100) * 100
                if quantity > 0:
                    self.place_market_order(symbol, OrderSide.BUY, quantity)
                    self.position_open = True
                    self.bars_since_entry = 0
        
        # デッドクロス（売りシグナル）
        elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and self.position_open:
            if self.bars_since_entry >= self.min_hold_bars:
                if position and position.quantity > 0:
                    self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                    self.position_open = False
                    self.bars_since_entry = 0


def convert_to_timeframe(bar_data, timeframe_minutes):
    """1分足データを指定した時間足に変換"""
    if timeframe_minutes == 1:
        return bar_data
    
    # リサンプリング
    bar_data_copy = bar_data.copy()
    bar_data_copy.set_index('time', inplace=True)
    
    resampled = bar_data_copy.resample(f'{timeframe_minutes}Min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled.reset_index(inplace=True)
    return resampled


def run_single_backtest(bar_data, timeframe, fast_ma, slow_ma, initial_capital=10_000_000):
    """単一のバックテストを実行"""
    # 指定時間足に変換
    tf_data = convert_to_timeframe(bar_data, timeframe)
    
    # ティック形式に変換（簡易版）
    tick_data = []
    for _, bar in tf_data.iterrows():
        tick_data.append({
            'time': bar['time'],
            'type': 'TRADE',
            'symbol': '7203 JT Equity',
            'value': bar['close'],
            'close': bar['close'],
            'size': bar['volume'] // 4
        })
    
    tick_df = pd.DataFrame(tick_data)
    
    # バックテスト実行
    strategy = OptimizedMAStrategy(
        fast_period=fast_ma,
        slow_period=slow_ma,
        min_hold_bars=1
    )
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=0.0003,
        slippage_rate=0.0001
    )
    
    engine.set_strategy(strategy)
    engine.run(tick_df)
    
    # メトリクス取得
    metrics = engine.get_performance_metrics()
    
    return {
        'timeframe': timeframe,
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'total_return': metrics.get('total_return', 0),
        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
        'max_drawdown': metrics.get('max_drawdown', 0),
        'total_trades': metrics.get('total_trades', 0),
        'win_rate': metrics.get('win_rate', 0),
        'final_equity': metrics.get('final_equity', initial_capital)
    }


def optimize_parameters():
    """パラメータの最適化を実行"""
    print("[INFO] トヨタ株 パラメータ最適化開始")
    
    # Bloomberg APIからデータ取得
    SYMBOL = "7203 JT Equity"
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=7)
    
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        # 1分足データ取得
        print("[INFO] データ取得中...")
        bar_data = fetcher.fetch_intraday_bars(SYMBOL, START_DATE, END_DATE, interval=1)
        
        if bar_data.empty:
            print("[ERROR] データが取得できませんでした")
            return
        
        print(f"[INFO] {len(bar_data)}本の1分足データを取得")
        
        # パラメータの組み合わせ
        timeframes = [1, 5, 15, 30, 60]  # 分
        fast_ma_periods = [5, 10, 15, 20, 25]
        slow_ma_periods = [20, 30, 40, 50, 60]
        
        # 結果を格納
        results = []
        total_combinations = len(timeframes) * len(fast_ma_periods) * len(slow_ma_periods)
        current = 0
        
        print(f"[INFO] {total_combinations}通りの組み合わせをテスト")
        print("[INFO] これには数分かかる場合があります...")
        
        # 全組み合わせをテスト
        for timeframe in timeframes:
            for fast_ma in fast_ma_periods:
                for slow_ma in slow_ma_periods:
                    if fast_ma >= slow_ma:
                        continue  # 無効な組み合わせをスキップ
                    
                    current += 1
                    if current % 10 == 0:
                        print(f"[INFO] 進捗: {current}/{total_combinations} ({current/total_combinations*100:.1f}%)")
                    
                    try:
                        result = run_single_backtest(
                            bar_data,
                            timeframe,
                            fast_ma,
                            slow_ma
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"[WARNING] エラー: TF={timeframe}, MA={fast_ma}/{slow_ma} - {e}")
                        continue
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)
        
        # 結果をソート（シャープレシオ順）
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        # 上位10件を表示
        print("\n[INFO] === 上位10件の結果（シャープレシオ順） ===")
        print("時間足(分) | MA期間 | リターン | シャープ | 最大DD | 取引数")
        print("-" * 60)
        
        for _, row in results_df.head(10).iterrows():
            print(f"{row['timeframe']:>10} | {row['fast_ma']:>3}/{row['slow_ma']:<3} | "
                  f"{row['total_return']:>7.2%} | {row['sharpe_ratio']:>6.2f} | "
                  f"{row['max_drawdown']:>6.2%} | {row['total_trades']:>6}")
        
        # 時間足別の最適パラメータ
        print("\n[INFO] === 時間足別の最適パラメータ ===")
        for tf in timeframes:
            tf_results = results_df[results_df['timeframe'] == tf]
            if not tf_results.empty:
                best = tf_results.iloc[0]
                print(f"{tf}分足: MA{best['fast_ma']}/{best['slow_ma']} "
                      f"(リターン: {best['total_return']:.2%}, シャープ: {best['sharpe_ratio']:.2f})")
        
        # 総合的な分析
        print("\n[INFO] === 総合分析 ===")
        
        # プラスリターンの組み合わせ
        positive_returns = results_df[results_df['total_return'] > 0]
        print(f"プラスリターン: {len(positive_returns)}/{len(results_df)} "
              f"({len(positive_returns)/len(results_df)*100:.1f}%)")
        
        # 時間足別の平均パフォーマンス
        print("\n時間足別平均パフォーマンス:")
        for tf in timeframes:
            tf_results = results_df[results_df['timeframe'] == tf]
            if not tf_results.empty:
                avg_return = tf_results['total_return'].mean()
                avg_sharpe = tf_results['sharpe_ratio'].mean()
                avg_trades = tf_results['total_trades'].mean()
                print(f"{tf:>3}分足: リターン {avg_return:>6.2%} | "
                      f"シャープ {avg_sharpe:>5.2f} | 取引数 {avg_trades:>5.1f}")
        
        # 最適な組み合わせの詳細
        best_result = results_df.iloc[0]
        print(f"\n[INFO] === 最適な組み合わせの詳細 ===")
        print(f"時間足: {best_result['timeframe']}分")
        print(f"移動平均期間: {best_result['fast_ma']}/{best_result['slow_ma']}")
        print(f"総リターン: {best_result['total_return']:.2%}")
        print(f"シャープレシオ: {best_result['sharpe_ratio']:.2f}")
        print(f"最大ドローダウン: {best_result['max_drawdown']:.2%}")
        print(f"総取引数: {best_result['total_trades']}")
        print(f"勝率: {best_result['win_rate']:.1%}")
        print(f"最終資産: ¥{best_result['final_equity']:,.0f}")
        
        # 結果をCSVに保存
        csv_filename = f"reports/toyota_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('reports', exist_ok=True)
        results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] 全結果を保存: {csv_filename}")
        
        # ヒートマップデータを作成
        create_heatmap_html(results_df)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()
        print("\n[INFO] 最適化完了")


def create_heatmap_html(results_df):
    """結果のヒートマップHTMLを作成"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>トヨタ株 パラメータ最適化結果</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .chart-container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>トヨタ自動車 (7203) - パラメータ最適化結果</h1>
        <p>移動平均クロス戦略の時間足とパラメータ最適化</p>
    </div>
"""
    
    # 各時間足のヒートマップを作成
    timeframes = sorted(results_df['timeframe'].unique())
    
    for i, tf in enumerate(timeframes):
        tf_data = results_df[results_df['timeframe'] == tf]
        
        # ヒートマップ用のピボットテーブル作成
        pivot_return = tf_data.pivot_table(
            values='total_return',
            index='slow_ma',
            columns='fast_ma',
            aggfunc='first'
        )
        
        pivot_sharpe = tf_data.pivot_table(
            values='sharpe_ratio',
            index='slow_ma',
            columns='fast_ma',
            aggfunc='first'
        )
        
        html_content += f"""
    <div class="chart-container">
        <h2>{tf}分足</h2>
        <div id="heatmap_return_{tf}" style="width:100%; height:400px;"></div>
        <div id="heatmap_sharpe_{tf}" style="width:100%; height:400px;"></div>
    </div>
    
    <script>
        // リターンのヒートマップ
        var data_return_{tf} = [{{
            z: {pivot_return.values.tolist()},
            x: {pivot_return.columns.tolist()},
            y: {pivot_return.index.tolist()},
            type: 'heatmap',
            colorscale: 'RdYlGn',
            zmid: 0,
            text: {pivot_return.values.tolist()},
            texttemplate: '%{{text:.1%}}',
            textfont: {{size: 10}},
            hovertemplate: 'Fast MA: %{{x}}<br>Slow MA: %{{y}}<br>Return: %{{z:.2%}}<extra></extra>'
        }}];
        
        var layout_return_{tf} = {{
            title: 'リターン (%)',
            xaxis: {{title: '短期MA期間'}},
            yaxis: {{title: '長期MA期間'}}
        }};
        
        Plotly.newPlot('heatmap_return_{tf}', data_return_{tf}, layout_return_{tf});
        
        // シャープレシオのヒートマップ
        var data_sharpe_{tf} = [{{
            z: {pivot_sharpe.values.tolist()},
            x: {pivot_sharpe.columns.tolist()},
            y: {pivot_sharpe.index.tolist()},
            type: 'heatmap',
            colorscale: 'Viridis',
            text: {pivot_sharpe.values.tolist()},
            texttemplate: '%{{text:.2f}}',
            textfont: {{size: 10}},
            hovertemplate: 'Fast MA: %{{x}}<br>Slow MA: %{{y}}<br>Sharpe: %{{z:.2f}}<extra></extra>'
        }}];
        
        var layout_sharpe_{tf} = {{
            title: 'シャープレシオ',
            xaxis: {{title: '短期MA期間'}},
            yaxis: {{title: '長期MA期間'}}
        }};
        
        Plotly.newPlot('heatmap_sharpe_{tf}', data_sharpe_{tf}, layout_sharpe_{tf});
    </script>
"""
    
    # 総合ランキング
    html_content += """
    <div class="chart-container">
        <h2>総合ランキング（上位20件）</h2>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background-color: #f8f9fa;">
                <th style="padding: 10px; border: 1px solid #ddd;">順位</th>
                <th style="padding: 10px; border: 1px solid #ddd;">時間足</th>
                <th style="padding: 10px; border: 1px solid #ddd;">MA期間</th>
                <th style="padding: 10px; border: 1px solid #ddd;">リターン</th>
                <th style="padding: 10px; border: 1px solid #ddd;">シャープ</th>
                <th style="padding: 10px; border: 1px solid #ddd;">最大DD</th>
                <th style="padding: 10px; border: 1px solid #ddd;">取引数</th>
            </tr>
"""
    
    for i, (_, row) in enumerate(results_df.head(20).iterrows()):
        color = '#e8f5e9' if row['total_return'] > 0 else '#ffebee'
        html_content += f"""
            <tr style="background-color: {color};">
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{i+1}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{row['timeframe']}分</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{row['fast_ma']}/{row['slow_ma']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{row['total_return']:.2%}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{row['sharpe_ratio']:.2f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{row['max_drawdown']:.2%}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{row['total_trades']}</td>
            </tr>
"""
    
    html_content += """
        </table>
    </div>
</body>
</html>
"""
    
    # ファイル保存
    filename = f"reports/toyota_optimization_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[INFO] ヒートマップを保存: {filename}")
    
    # ブラウザで開く
    import webbrowser
    webbrowser.open(f'file:///{os.path.abspath(filename)}')


if __name__ == "__main__":
    optimize_parameters()