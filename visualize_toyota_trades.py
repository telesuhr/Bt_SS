"""
トヨタのバックテスト結果をチャートで可視化
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
from src.backtest.engine import BacktestEngine, OrderSide
from backtest_toyota_bars import SimpleMAStrategy, convert_bars_to_ticks


def visualize_backtest_results():
    """バックテスト結果を可視化"""
    print("[INFO] トヨタ バックテスト結果の可視化")
    
    # Bloomberg APIから分足データ取得
    SYMBOL = "7203 JT Equity"
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=7)
    
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        # データ取得
        print("[INFO] データ取得中...")
        bar_data = fetcher.fetch_intraday_bars(SYMBOL, START_DATE, END_DATE, interval=1)
        
        if bar_data.empty:
            print("[ERROR] データが取得できませんでした")
            return
        
        print(f"[INFO] {len(bar_data)}本のバーデータを取得")
        
        # バックテスト実行
        tick_data = convert_bars_to_ticks(bar_data)
        
        # 戦略とエンジン設定
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30)
        engine = BacktestEngine(
            initial_capital=10_000_000,
            commission_rate=0.0003,
            slippage_rate=0.0001
        )
        
        engine.set_strategy(strategy)
        engine.run(tick_data)
        
        # 移動平均を計算
        bar_data['MA_fast'] = bar_data['close'].rolling(window=10).mean()
        bar_data['MA_slow'] = bar_data['close'].rolling(window=30).mean()
        
        # 図の作成（3つのサブプロット）
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                             gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. 価格チャートと移動平均
        ax1.plot(bar_data['time'], bar_data['close'], 'b-', linewidth=0.5, alpha=0.7, label='終値')
        ax1.plot(bar_data['time'], bar_data['MA_fast'], 'r-', linewidth=1.5, label='MA(10)')
        ax1.plot(bar_data['time'], bar_data['MA_slow'], 'g-', linewidth=1.5, label='MA(30)')
        
        # 取引をプロット
        buy_trades = []
        sell_trades = []
        
        for trade in engine.trades:
            if trade.side == OrderSide.BUY:
                buy_trades.append((trade.timestamp, trade.price))
            else:
                sell_trades.append((trade.timestamp, trade.price))
        
        if buy_trades:
            buy_times, buy_prices = zip(*buy_trades)
            ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, 
                       label=f'買い ({len(buy_trades)}回)', zorder=5, alpha=0.8)
        
        if sell_trades:
            sell_times, sell_prices = zip(*sell_trades)
            ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, 
                       label=f'売り ({len(sell_trades)}回)', zorder=5, alpha=0.8)
        
        ax1.set_title(f'トヨタ自動車 (7203) - 価格チャートと取引タイミング\n'
                      f'期間: {START_DATE.strftime("%Y/%m/%d")} - {END_DATE.strftime("%Y/%m/%d")}',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('価格 (円)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. ボリューム
        ax2.bar(bar_data['time'], bar_data['volume'], width=0.0007, alpha=0.5, color='gray')
        ax2.set_ylabel('出来高', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. エクイティカーブ
        equity_df = pd.DataFrame(engine.equity_curve)
        ax3.plot(equity_df['timestamp'], equity_df['equity'], 'b-', linewidth=2)
        ax3.axhline(y=engine.initial_capital, color='gray', linestyle='--', alpha=0.7)
        ax3.fill_between(equity_df['timestamp'], engine.initial_capital, equity_df['equity'],
                        where=equity_df['equity'] < engine.initial_capital,
                        color='red', alpha=0.3, interpolate=True)
        ax3.fill_between(equity_df['timestamp'], engine.initial_capital, equity_df['equity'],
                        where=equity_df['equity'] >= engine.initial_capital,
                        color='green', alpha=0.3, interpolate=True)
        
        ax3.set_ylabel('資産総額 (円)', fontsize=12)
        ax3.set_xlabel('時刻', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # x軸のフォーマット
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # パフォーマンスメトリクスを追加
        metrics = engine.get_performance_metrics()
        textstr = f'総リターン: {metrics.get("total_return", 0):.2%}\n' \
                  f'シャープレシオ: {metrics.get("sharpe_ratio", 0):.2f}\n' \
                  f'最大DD: {metrics.get("max_drawdown", 0):.2%}\n' \
                  f'総取引数: {metrics.get("total_trades", 0)}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # 保存
        filename = f'reports/toyota_trades_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        os.makedirs('reports', exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"[INFO] チャートを保存: {filename}")
        
        # 追加の分析チャート
        create_detailed_analysis(bar_data, engine, strategy)
        
        plt.show()
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()


def create_detailed_analysis(bar_data, engine, strategy):
    """詳細な分析チャートを作成"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 取引の分布（時間帯別）
    trades_df = pd.DataFrame([vars(trade) for trade in engine.trades])
    trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
    
    trade_counts = trades_df.groupby(['hour', 'side']).size().unstack(fill_value=0)
    if not trade_counts.empty:
        trade_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
        ax1.set_title('時間帯別の取引回数', fontsize=12)
        ax1.set_xlabel('時刻')
        ax1.set_ylabel('取引回数')
        ax1.legend(['買い', '売り'])
    
    # 2. リターンのヒストグラム
    equity_df = pd.DataFrame(engine.equity_curve)
    returns = equity_df['equity'].pct_change().dropna()
    ax2.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax2.set_title('リターン分布', fontsize=12)
    ax2.set_xlabel('リターン')
    ax2.set_ylabel('頻度')
    
    # 3. 移動平均の乖離
    bar_data['MA_diff'] = bar_data['MA_fast'] - bar_data['MA_slow']
    bar_data['MA_diff_pct'] = (bar_data['MA_diff'] / bar_data['MA_slow'] * 100)
    
    ax3.plot(bar_data['time'], bar_data['MA_diff_pct'], 'b-', linewidth=1)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax3.fill_between(bar_data['time'], 0, bar_data['MA_diff_pct'],
                     where=bar_data['MA_diff_pct'] > 0, color='green', alpha=0.3)
    ax3.fill_between(bar_data['time'], 0, bar_data['MA_diff_pct'],
                     where=bar_data['MA_diff_pct'] <= 0, color='red', alpha=0.3)
    ax3.set_title('移動平均の乖離率 (MA10 - MA30)', fontsize=12)
    ax3.set_xlabel('時刻')
    ax3.set_ylabel('乖離率 (%)')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 4. ドローダウン
    equity_df['drawdown'] = (equity_df['equity'] / equity_df['equity'].cummax() - 1) * 100
    ax4.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0,
                     color='red', alpha=0.5)
    ax4.plot(equity_df['timestamp'], equity_df['drawdown'], 'r-', linewidth=1)
    ax4.set_title('ドローダウン', fontsize=12)
    ax4.set_xlabel('時刻')
    ax4.set_ylabel('ドローダウン (%)')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # x軸の回転
    for ax in [ax3, ax4]:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # 保存
    filename = f'reports/toyota_detailed_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[INFO] 詳細分析チャートを保存: {filename}")


if __name__ == "__main__":
    visualize_backtest_results()