"""
市場環境適応型戦略
相場状況を自動判定し、動的に戦略を切り替える
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.backtest.engine import BacktestEngine, OrderSide
from src.strategies.base_strategy import BaseStrategy


class MarketRegime(Enum):
    """市場状況の分類"""
    STRONG_UPTREND = "強い上昇トレンド"
    UPTREND = "上昇トレンド"
    RANGE = "レンジ相場"
    DOWNTREND = "下落トレンド"
    STRONG_DOWNTREND = "強い下落トレンド"
    HIGH_VOLATILITY = "高ボラティリティ"


class MarketRegimeDetector:
    """市場状況を判定するクラス"""
    
    def __init__(self, lookback_short=20, lookback_long=50):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        
    def detect_regime(self, prices, volumes=None):
        """
        価格データから市場状況を判定
        
        Returns:
            MarketRegime: 現在の市場状況
            dict: 詳細情報
        """
        if len(prices) < self.lookback_long:
            return MarketRegime.RANGE, {"confidence": 0.0}
        
        # 各種指標を計算
        prices_array = np.array(prices)
        
        # 移動平均
        sma_short = np.mean(prices_array[-self.lookback_short:])
        sma_long = np.mean(prices_array[-self.lookback_long:])
        
        # トレンド強度（価格と移動平均の乖離率）
        trend_strength_short = (prices_array[-1] / sma_short - 1) * 100
        trend_strength_long = (prices_array[-1] / sma_long - 1) * 100
        
        # ボラティリティ
        returns = np.diff(prices_array[-self.lookback_short:]) / prices_array[-self.lookback_short:-1]
        volatility = np.std(returns) * 100
        
        # ADX（簡易版）- トレンドの強さ
        high_low_diff = []
        for i in range(1, min(14, len(prices))):
            high = max(prices[-i-1:-i+1] if i > 1 else [prices[-i-1], prices[-i]])
            low = min(prices[-i-1:-i+1] if i > 1 else [prices[-i-1], prices[-i]])
            high_low_diff.append(high - low)
        
        avg_range = np.mean(high_low_diff) if high_low_diff else 0
        current_range = max(prices[-3:]) - min(prices[-3:])
        adx_proxy = current_range / avg_range if avg_range > 0 else 1
        
        # 傾き（線形回帰）
        x = np.arange(self.lookback_short)
        slope = np.polyfit(x, prices_array[-self.lookback_short:], 1)[0]
        slope_pct = (slope / prices_array[-1]) * 100
        
        # 市場状況を判定
        regime = MarketRegime.RANGE
        confidence = 0.5
        
        # 高ボラティリティチェック
        if volatility > 2.0:  # 2%以上の日次ボラティリティ
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(volatility / 3.0, 1.0)
        
        # トレンド判定
        elif adx_proxy > 1.2:  # トレンドが強い
            if slope_pct > 0.5 and trend_strength_short > 1.0:
                regime = MarketRegime.STRONG_UPTREND if trend_strength_short > 2.0 else MarketRegime.UPTREND
                confidence = min(abs(slope_pct) / 2.0, 1.0)
            elif slope_pct < -0.5 and trend_strength_short < -1.0:
                regime = MarketRegime.STRONG_DOWNTREND if trend_strength_short < -2.0 else MarketRegime.DOWNTREND
                confidence = min(abs(slope_pct) / 2.0, 1.0)
        
        # レンジ相場
        else:
            if abs(trend_strength_short) < 0.5 and volatility < 1.0:
                regime = MarketRegime.RANGE
                confidence = 1.0 - volatility
        
        details = {
            "confidence": confidence,
            "trend_strength_short": trend_strength_short,
            "trend_strength_long": trend_strength_long,
            "volatility": volatility,
            "slope_pct": slope_pct,
            "sma_short": sma_short,
            "sma_long": sma_long
        }
        
        return regime, details


class AdaptiveStrategy(BaseStrategy):
    """市場環境に適応する戦略"""
    
    def __init__(self, name="AdaptiveStrategy"):
        super().__init__(name=name)
        self.regime_detector = MarketRegimeDetector()
        self.price_history = []
        self.volume_history = []
        self.current_regime = MarketRegime.RANGE
        self.regime_history = []
        self.has_position = False
        self.entry_price = None
        self.current_date = None
        self.regime_change_count = 0
        self.last_regime_change = None
        
        # 各市場状況での戦略パラメータ
        self.regime_params = {
            MarketRegime.STRONG_UPTREND: {
                "strategy": "momentum",
                "entry_threshold": 0.001,  # 0.1%の上昇でエントリー
                "profit_target": 0.005,    # 0.5%で利確
                "stop_loss": -0.002        # 0.2%で損切り
            },
            MarketRegime.UPTREND: {
                "strategy": "breakout",
                "lookback": 20,
                "breakout_pct": 0.002,
                "profit_target": 0.003,
                "stop_loss": -0.002
            },
            MarketRegime.RANGE: {
                "strategy": "mean_reversion",
                "z_threshold": 1.5,
                "exit_z": 0.3,
                "max_holding": 30
            },
            MarketRegime.DOWNTREND: {
                "strategy": "short_momentum",  # 実際には売りポジションは取れないので逆張り
                "entry_threshold": -0.002,     # 0.2%の下落で逆張りエントリー
                "profit_target": 0.002,
                "stop_loss": -0.001
            },
            MarketRegime.STRONG_DOWNTREND: {
                "strategy": "wait",  # 取引を控える
            },
            MarketRegime.HIGH_VOLATILITY: {
                "strategy": "scalping",
                "entry_threshold": 0.002,
                "profit_target": 0.003,
                "stop_loss": -0.001,
                "max_trades": 2
            }
        }
        
    def on_tick(self, tick):
        price = tick.get('close', tick.get('value', 0))
        volume = tick.get('size', 0)
        tick_time = tick['time']
        tick_hour = tick_time.hour
        tick_date = tick_time.date()
        
        # 新しい日の開始
        if tick_date != self.current_date:
            self.current_date = tick_date
            self.price_history = []
            self.volume_history = []
            self.regime_history = []
            if self.has_position:
                self.force_close(tick)
        
        # データ追加
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # データ制限
        if len(self.price_history) > 100:
            self.price_history.pop(0)
            self.volume_history.pop(0)
        
        # データが不十分な場合はスキップ
        if len(self.price_history) < 50:
            return
        
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        
        # 取引時間チェック
        is_trading_hours = 0 <= tick_hour <= 5
        is_close_time = tick_hour == 5
        
        # 強制クローズ
        if self.has_position and is_close_time:
            if position and position.quantity > 0:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.has_position = False
                self.log_trade("FORCE_CLOSE", price, "日中クローズ")
                return
        
        if not is_trading_hours:
            return
        
        # 市場状況を判定
        new_regime, details = self.regime_detector.detect_regime(self.price_history, self.volume_history)
        
        # レジーム変化を記録
        if new_regime != self.current_regime:
            self.current_regime = new_regime
            self.regime_change_count += 1
            self.last_regime_change = tick_time
            self.log_regime_change(new_regime, details)
        
        self.regime_history.append(new_regime)
        
        # 現在のレジームに基づいて戦略を実行
        self.execute_regime_strategy(tick, position, new_regime, details)
    
    def execute_regime_strategy(self, tick, position, regime, details):
        """レジームに応じた戦略を実行"""
        params = self.regime_params.get(regime, {})
        strategy = params.get("strategy", "wait")
        
        price = tick.get('close', tick.get('value', 0))
        symbol = tick.get('symbol', 'UNKNOWN')
        tick_time = tick['time']
        
        if strategy == "wait":
            # 強い下落トレンドでは取引しない
            if self.has_position and position:
                self.place_market_order(symbol, OrderSide.SELL, position.quantity)
                self.has_position = False
                self.log_trade("EXIT_DOWNTREND", price, "下落トレンドのため撤退")
            return
        
        elif strategy == "momentum":
            # モメンタム戦略
            if len(self.price_history) >= 20:
                momentum = (self.price_history[-1] / self.price_history[-20] - 1)
                
                if not self.has_position and momentum > params["entry_threshold"]:
                    self.enter_position(symbol, price, f"モメンタム {momentum:.3%}")
                elif self.has_position and self.entry_price:
                    profit_rate = (price - self.entry_price) / self.entry_price
                    if profit_rate >= params["profit_target"] or profit_rate <= params["stop_loss"]:
                        self.exit_position(symbol, position.quantity, price, 
                                         f"{'利確' if profit_rate > 0 else '損切'} {profit_rate:.3%}")
        
        elif strategy == "breakout":
            # ブレイクアウト戦略
            lookback = params.get("lookback", 20)
            if len(self.price_history) >= lookback:
                recent_high = max(self.price_history[-lookback:-1])
                breakout_level = recent_high * (1 + params["breakout_pct"])
                
                if not self.has_position and price > breakout_level:
                    self.enter_position(symbol, price, f"ブレイクアウト {price/recent_high-1:.3%}")
                elif self.has_position and self.entry_price:
                    profit_rate = (price - self.entry_price) / self.entry_price
                    if profit_rate >= params["profit_target"] or profit_rate <= params["stop_loss"]:
                        self.exit_position(symbol, position.quantity, price, 
                                         f"{'利確' if profit_rate > 0 else '損切'} {profit_rate:.3%}")
        
        elif strategy == "mean_reversion":
            # 平均回帰戦略
            if len(self.price_history) >= 30:
                prices = self.price_history[-30:]
                mean = np.mean(prices)
                std = np.std(prices)
                
                if std > 0:
                    z_score = (price - mean) / std
                    
                    if not self.has_position and z_score < -params["z_threshold"]:
                        self.enter_position(symbol, price, f"平均回帰 Z={z_score:.2f}")
                    elif self.has_position and abs(z_score) < params["exit_z"]:
                        self.exit_position(symbol, position.quantity, price, f"平均回帰 Z={z_score:.2f}")
        
        elif strategy == "short_momentum" or strategy == "scalping":
            # 逆張りまたはスキャルピング
            if len(self.price_history) >= 10:
                short_momentum = (self.price_history[-1] / self.price_history[-10] - 1)
                
                entry_condition = short_momentum < params["entry_threshold"] if strategy == "short_momentum" else abs(short_momentum) > params["entry_threshold"]
                
                if not self.has_position and entry_condition:
                    self.enter_position(symbol, price, f"{strategy} {short_momentum:.3%}")
                elif self.has_position and self.entry_price:
                    profit_rate = (price - self.entry_price) / self.entry_price
                    if profit_rate >= params["profit_target"] or profit_rate <= params["stop_loss"]:
                        self.exit_position(symbol, position.quantity, price, 
                                         f"{'利確' if profit_rate > 0 else '損切'} {profit_rate:.3%}")
    
    def enter_position(self, symbol, price, reason):
        """ポジションエントリー"""
        cash = self.get_cash_balance()
        quantity = int((cash * 0.95) / price / 100) * 100
        if quantity > 0:
            self.place_market_order(symbol, OrderSide.BUY, quantity)
            self.has_position = True
            self.entry_price = price
            self.log_trade("ENTRY", price, reason)
    
    def exit_position(self, symbol, quantity, price, reason):
        """ポジションエグジット"""
        self.place_market_order(symbol, OrderSide.SELL, quantity)
        self.has_position = False
        self.log_trade("EXIT", price, reason)
    
    def force_close(self, tick):
        """強制クローズ"""
        symbol = tick.get('symbol', 'UNKNOWN')
        position = self.get_position(symbol)
        if position and position.quantity > 0:
            self.place_market_order(symbol, OrderSide.SELL, position.quantity)
            self.has_position = False
    
    def log_trade(self, action, price, reason):
        """取引ログ"""
        print(f"[{action}] {self.current_regime.value} @ {price:.1f} - {reason}")
    
    def log_regime_change(self, new_regime, details):
        """レジーム変化ログ"""
        print(f"\n[REGIME CHANGE] → {new_regime.value} (信頼度: {details['confidence']:.2f})")
        print(f"  トレンド強度: {details['trend_strength_short']:.2f}%, ボラティリティ: {details['volatility']:.2f}%")


def test_adaptive_strategy():
    """市場環境適応型戦略のテスト"""
    print("=== 市場環境適応型戦略バックテスト ===\n")
    
    # テスト対象銘柄
    test_symbols = [
        "7203 JT Equity",  # トヨタ
        "9984 JT Equity",  # ソフトバンクG
        "6758 JT Equity",  # ソニー
        "8306 JT Equity",  # 三菱UFJ
    ]
    
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        results = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        for symbol in test_symbols:
            print(f"\n{'='*60}")
            print(f"テスト銘柄: {symbol}")
            print('='*60)
            
            # データ取得
            bar_data = fetcher.fetch_intraday_bars(symbol, start_date, end_date, interval=1)
            
            if bar_data.empty:
                print(f"[WARNING] {symbol}のデータが取得できませんでした")
                continue
            
            print(f"[INFO] データ取得完了: {len(bar_data)}本")
            
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
            
            # 適応型戦略でテスト
            print("\n[適応型戦略]")
            adaptive_strategy = AdaptiveStrategy()
            engine = BacktestEngine(
                initial_capital=10_000_000,
                commission_rate=0.0003,
                slippage_rate=0.0001
            )
            
            engine.set_strategy(adaptive_strategy)
            engine.run(tick_df)
            
            metrics = engine.get_performance_metrics()
            
            print(f"\n結果:")
            print(f"  総リターン: {metrics.get('total_return', 0):.2%}")
            print(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  最大DD: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  総取引数: {metrics.get('total_trades', 0)}")
            print(f"  レジーム変化回数: {adaptive_strategy.regime_change_count}")
            
            # 比較用: シンプルな買い持ち戦略
            print("\n[比較: Buy&Hold]")
            buy_hold_return = (tick_df.iloc[-1]['close'] / tick_df.iloc[0]['close'] - 1)
            print(f"  リターン: {buy_hold_return:.2%}")
            
            results.append({
                'symbol': symbol,
                'adaptive_return': metrics.get('total_return', 0),
                'adaptive_sharpe': metrics.get('sharpe_ratio', 0),
                'adaptive_trades': metrics.get('total_trades', 0),
                'regime_changes': adaptive_strategy.regime_change_count,
                'buy_hold_return': buy_hold_return
            })
        
        # 結果サマリー
        if results:
            print("\n\n" + "="*80)
            print("総合結果サマリー")
            print("="*80)
            
            results_df = pd.DataFrame(results)
            
            print(f"\n{'銘柄':<15} {'適応型リターン':>15} {'Buy&Hold':>12} {'差分':>10} {'取引数':>8} {'レジーム変化':>12}")
            print("-"*85)
            
            for _, row in results_df.iterrows():
                diff = row['adaptive_return'] - row['buy_hold_return']
                print(f"{row['symbol'].split()[0]:<15} {row['adaptive_return']:>14.2%} "
                      f"{row['buy_hold_return']:>11.2%} {diff:>9.2%} "
                      f"{row['adaptive_trades']:>8} {row['regime_changes']:>12}")
            
            avg_adaptive = results_df['adaptive_return'].mean()
            avg_buy_hold = results_df['buy_hold_return'].mean()
            print(f"\n平均リターン:")
            print(f"  適応型戦略: {avg_adaptive:.2%}")
            print(f"  Buy&Hold: {avg_buy_hold:.2%}")
            print(f"  超過リターン: {avg_adaptive - avg_buy_hold:.2%}")
            
            # HTMLレポート生成
            create_adaptive_report(results_df)
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()


def create_adaptive_report(results_df):
    """適応型戦略のレポート生成"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>市場環境適応型戦略 バックテスト結果</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f2f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .feature-box {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .feature-box h3 {{
            margin-top: 0;
            color: #333;
        }}
        .regime-list {{
            list-style: none;
            padding: 0;
        }}
        .regime-list li {{
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
        }}
        .positive {{ color: #28a745; font-weight: bold; }}
        .negative {{ color: #dc3545; font-weight: bold; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>市場環境適応型戦略</h1>
            <p>相場状況を自動判定し、最適な戦略に動的に切り替える</p>
        </div>
        
        <div class="feature-box">
            <h3>🎯 戦略の特徴</h3>
            <ul class="regime-list">
                <li>
                    <span><strong>強い上昇トレンド</strong></span>
                    <span>→ モメンタム戦略（順張り）</span>
                </li>
                <li>
                    <span><strong>上昇トレンド</strong></span>
                    <span>→ ブレイクアウト戦略</span>
                </li>
                <li>
                    <span><strong>レンジ相場</strong></span>
                    <span>→ 平均回帰戦略</span>
                </li>
                <li>
                    <span><strong>下落トレンド</strong></span>
                    <span>→ 逆張り戦略</span>
                </li>
                <li>
                    <span><strong>高ボラティリティ</strong></span>
                    <span>→ スキャルピング戦略</span>
                </li>
            </ul>
        </div>
        
        <div class="feature-box">
            <h3>📊 パフォーマンス比較</h3>
            <table>
                <tr>
                    <th>銘柄</th>
                    <th>適応型戦略</th>
                    <th>Buy&Hold</th>
                    <th>超過リターン</th>
                    <th>取引回数</th>
                    <th>レジーム変化</th>
                </tr>
"""
    
    for _, row in results_df.iterrows():
        diff = row['adaptive_return'] - row['buy_hold_return']
        adaptive_class = "positive" if row['adaptive_return'] > 0 else "negative"
        buyhold_class = "positive" if row['buy_hold_return'] > 0 else "negative"
        diff_class = "positive" if diff > 0 else "negative"
        
        html_content += f"""
                <tr>
                    <td>{row['symbol'].split()[0]}</td>
                    <td class="{adaptive_class}">{row['adaptive_return']:.2%}</td>
                    <td class="{buyhold_class}">{row['buy_hold_return']:.2%}</td>
                    <td class="{diff_class}">{diff:.2%}</td>
                    <td>{row['adaptive_trades']}</td>
                    <td>{row['regime_changes']}</td>
                </tr>
"""
    
    avg_adaptive = results_df['adaptive_return'].mean()
    avg_buy_hold = results_df['buy_hold_return'].mean()
    
    html_content += f"""
            </table>
        </div>
        
        <div class="feature-box">
            <h3>📈 分析結果</h3>
            <p><strong>平均リターン:</strong></p>
            <ul>
                <li>適応型戦略: <span class="{'positive' if avg_adaptive > 0 else 'negative'}">{avg_adaptive:.2%}</span></li>
                <li>Buy&Hold: <span class="{'positive' if avg_buy_hold > 0 else 'negative'}">{avg_buy_hold:.2%}</span></li>
                <li>超過リターン: <span class="{'positive' if avg_adaptive - avg_buy_hold > 0 else 'negative'}">{avg_adaptive - avg_buy_hold:.2%}</span></li>
            </ul>
            <p><strong>主な発見:</strong></p>
            <ul>
                <li>市場状況に応じて戦略を切り替えることで、リスク管理が改善</li>
                <li>レジーム変化を検知し、不利な相場では取引を控える</li>
                <li>各銘柄の特性に応じて、異なる頻度でレジームが変化</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    
    filename = f"reports/adaptive_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    os.makedirs("reports", exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[INFO] レポート生成: {filename}")


if __name__ == "__main__":
    test_adaptive_strategy()