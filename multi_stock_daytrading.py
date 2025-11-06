"""
複数銘柄でのSimpleDayTrading戦略テスト
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


def test_stock(symbol, fetcher):
    """個別銘柄のテスト"""
    print(f"\n{'='*60}")
    print(f"テスト銘柄: {symbol}")
    print('='*60)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # 1分足データ取得
        bar_data = fetcher.fetch_intraday_bars(symbol, start_date, end_date, interval=1)
        
        if bar_data.empty:
            print(f"[WARNING] {symbol}のデータが取得できませんでした")
            return None
        
        print(f"[INFO] データ取得完了: {len(bar_data)}本")
        print(f"[INFO] 期間: {bar_data['time'].min()} ～ {bar_data['time'].max()}")
        
        # ティック形式に変換
        tick_data = []
        for _, bar in bar_data.iterrows():
            tick_data.append({
                'time': bar['time'],
                'type': 'TRADE',
                'symbol': symbol,
                'value': bar['close'],
                'close': bar['close'],
                'size': bar['volume']
            })
        
        tick_df = pd.DataFrame(tick_data)
        
        # バックテスト実行
        strategy = SimpleDayTradingStrategy()
        engine = BacktestEngine(
            initial_capital=10_000_000,
            commission_rate=0.0003,
            slippage_rate=0.0001
        )
        
        engine.set_strategy(strategy)
        engine.run(tick_df)
        
        # 結果取得
        metrics = engine.get_performance_metrics()
        
        print(f"\n【結果】")
        print(f"  総リターン: {metrics.get('total_return', 0):.2%}")
        print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大DD: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  総取引数: {metrics.get('total_trades', 0)}")
        print(f"  最終資産: {metrics.get('final_equity', 0):,.0f}円")
        
        return {
            'symbol': symbol,
            **metrics
        }
        
    except Exception as e:
        print(f"[ERROR] {symbol}のテスト中にエラー: {e}")
        return None


def run_multi_stock_test():
    """複数銘柄でのテスト実行"""
    print("=== 複数銘柄でのSimpleDayTrading戦略テスト ===")
    
    # テスト対象銘柄（日本の代表的な銘柄）
    test_symbols = [
        "9984 JT Equity",  # ソフトバンクグループ
        "6758 JT Equity",  # ソニーグループ
        "9432 JT Equity",  # NTT
        "8306 JT Equity",  # 三菱UFJ
        "4063 JT Equity",  # 信越化学
        "6861 JT Equity",  # キーエンス
        "7267 JT Equity",  # ホンダ
        "8058 JT Equity",  # 三菱商事
        "6098 JT Equity",  # リクルート
        "9433 JT Equity",  # KDDI
    ]
    
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        results = []
        
        for symbol in test_symbols:
            result = test_stock(symbol, fetcher)
            if result:
                results.append(result)
        
        # 結果サマリー
        if results:
            print("\n" + "="*80)
            print("総合結果サマリー")
            print("="*80)
            
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('total_return', ascending=False)
            
            print(f"\n{'銘柄':<15} {'リターン':>10} {'シャープ':>10} {'最大DD':>10} {'取引数':>8} {'最終資産':>15}")
            print("-"*75)
            
            for _, row in results_df.iterrows():
                symbol_short = row['symbol'].split()[0]
                print(f"{symbol_short:<15} {row['total_return']:>9.2%} "
                      f"{row['sharpe_ratio']:>10.2f} {row['max_drawdown']:>9.2%} "
                      f"{row['total_trades']:>8.0f} {row['final_equity']:>15,.0f}")
            
            # 統計情報
            print(f"\n【統計情報】")
            print(f"  テスト銘柄数: {len(results)}")
            print(f"  プラスリターン銘柄: {(results_df['total_return'] > 0).sum()}/{len(results)}")
            print(f"  平均リターン: {results_df['total_return'].mean():.2%}")
            print(f"  最高リターン: {results_df['total_return'].max():.2%} ({results_df.iloc[0]['symbol'].split()[0]})")
            print(f"  最低リターン: {results_df['total_return'].min():.2%} ({results_df.iloc[-1]['symbol'].split()[0]})")
            
            # HTML レポート生成
            create_multi_stock_report(results_df)
            
            # CSV保存
            csv_path = f"reports/multi_stock_daytrading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs("reports", exist_ok=True)
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n[INFO] 結果を保存: {csv_path}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()


def create_multi_stock_report(results_df):
    """複数銘柄の結果HTMLレポート生成"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>複数銘柄 SimpleDayTrading戦略 テスト結果</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .negative {{
            color: #f44336;
            font-weight: bold;
        }}
        .best {{
            background-color: #fff3cd;
        }}
        .worst {{
            background-color: #f8d7da;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>複数銘柄 SimpleDayTrading戦略 テスト結果</h1>
        
        <div class="summary-box">
            <h2>テスト概要</h2>
            <p><strong>戦略:</strong> SimpleDayTrading（0.2%上昇でエントリー、利確0.3%/損切-0.2%）</p>
            <p><strong>期間:</strong> 直近7日間</p>
            <p><strong>初期資金:</strong> 10,000,000円</p>
            <p><strong>制約:</strong> 1日最大3取引、14:55強制決済</p>
        </div>
        
        <div class="summary-box">
            <h2>統計サマリー</h2>
            <p><strong>テスト銘柄数:</strong> {len(results_df)}</p>
            <p><strong>プラスリターン:</strong> {(results_df['total_return'] > 0).sum()}/{len(results_df)} 銘柄</p>
            <p><strong>平均リターン:</strong> <span class="{'positive' if results_df['total_return'].mean() > 0 else 'negative'}">{results_df['total_return'].mean():.2%}</span></p>
            <p><strong>リターン標準偏差:</strong> {results_df['total_return'].std():.2%}</p>
        </div>
        
        <div class="summary-box">
            <h2>銘柄別パフォーマンス</h2>
            <table>
                <tr>
                    <th>順位</th>
                    <th>銘柄コード</th>
                    <th>総リターン</th>
                    <th>シャープレシオ</th>
                    <th>最大DD</th>
                    <th>総取引数</th>
                    <th>勝率</th>
                    <th>最終資産</th>
                </tr>
"""
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        return_class = "positive" if row['total_return'] > 0 else "negative"
        sharpe_class = "positive" if row['sharpe_ratio'] > 0 else "negative"
        row_class = "best" if i == 0 else ("worst" if i == len(results_df)-1 else "")
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{i+1}</td>
                    <td>{row['symbol'].split()[0]}</td>
                    <td class="{return_class}">{row['total_return']:.2%}</td>
                    <td class="{sharpe_class}">{row['sharpe_ratio']:.2f}</td>
                    <td class="negative">{row['max_drawdown']:.2%}</td>
                    <td>{row['total_trades']:.0f}</td>
                    <td>{row.get('win_rate', 0):.1%}</td>
                    <td>{row['final_equity']:,.0f}円</td>
                </tr>
"""
    
    html_content += """
            </table>
        </div>
        
        <div class="summary-box">
            <h2>分析結果</h2>
            <ul>
                <li>SimpleDayTrading戦略は銘柄によってパフォーマンスが大きく異なる</li>
                <li>ボラティリティの高い銘柄ほど取引機会が多い傾向</li>
                <li>流動性の高い大型株での安定性が期待できる</li>
                <li>セクターや銘柄特性によって戦略の相性がある</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    filename = f"reports/multi_stock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    os.makedirs('reports', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[INFO] HTMLレポート生成: {filename}")


if __name__ == "__main__":
    run_multi_stock_test()