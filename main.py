import os
import sys
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """ロギング設定"""
    logger.remove()
    logger.add(
        "logs/backtest_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.add(sys.stdout, level="INFO")


# モジュールのインポート
from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.data.storage import DataStorage

# TA-Libが利用可能か確認
try:
    from src.technical_indicators.indicators import TechnicalIndicators
except ImportError:
    logger.warning("TA-Lib not found, using lite version")
    from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators

from src.backtest.engine import BacktestEngine
from src.strategies.base_strategy import TickMomentumStrategy, SpreadArbitrageStrategy, VWAPStrategy
from src.analysis.analyzer import BacktestAnalyzer


def run_backtest_example():
    """バックテスト実行例"""
    
    # 環境変数読み込み
    load_dotenv()
    
    # パラメータ設定
    SYMBOL = "7203 JT Equity"  # トヨタ自動車
    START_DATE = datetime(2024, 11, 7, 9, 0, 0)  # 2024/11/7 9:00
    END_DATE = datetime(2024, 11, 7, 15, 0, 0)   # 2024/11/7 15:00
    
    logger.info("=== ティックデータバックテストシステム ===")
    logger.info(f"対象銘柄: {SYMBOL}")
    logger.info(f"期間: {START_DATE} ～ {END_DATE}")
    
    # 1. データ取得
    USE_BLOOMBERG = os.getenv('USE_BLOOMBERG', 'false').lower() == 'true'
    
    if USE_BLOOMBERG:
        logger.info("1. Bloomberg APIからデータ取得")
        try:
            fetcher = TickDataFetcher()
            if fetcher.connect():
                tick_data = fetcher.fetch_tick_data(SYMBOL, START_DATE, END_DATE)
                if not tick_data.empty:
                    logger.info(f"取得データ: {len(tick_data)}ティック")
                else:
                    tick_data = None
            else:
                tick_data = None
        except Exception as e:
            logger.warning(f"Bloomberg API使用不可: {e}")
            tick_data = None
    else:
        tick_data = None
    
    if tick_data is None or tick_data.empty:
        logger.info("デモデータを生成します")
        tick_data = generate_demo_tick_data(SYMBOL, START_DATE, END_DATE)
    
    logger.info(f"取得データ: {len(tick_data)}ティック")
    
    # 2. データ保存
    USE_DATABASE = os.getenv('USE_DATABASE', 'false').lower() == 'true'
    
    if USE_DATABASE:
        try:
            logger.info("2. データをストレージに保存")
            storage = DataStorage()
            storage.save_tick_data(tick_data, SYMBOL)
        except Exception as e:
            logger.warning(f"データ保存エラー: {e}")
            storage = None
    else:
        storage = None
        logger.info("2. データ保存をスキップ（メモリ上で処理）")
    
    # 3. テクニカル指標計算
    logger.info("3. テクニカル指標を計算")
    indicators = TechnicalIndicators()
    
    # OHLCデータ生成
    ohlc_data = indicators.tick_to_ohlc(tick_data, interval='1Min')
    logger.info(f"生成された1分足データ: {len(ohlc_data)}本")
    
    # 各種指標計算
    ohlc_data['sma_20'] = indicators.sma(ohlc_data['close'], 20)
    ohlc_data['rsi'] = indicators.rsi(ohlc_data['close'], 14)
    
    # 4. バックテスト実行
    logger.info("4. バックテスト実行")
    
    # 複数の戦略でテスト
    strategies = [
        TickMomentumStrategy(lookback_ticks=50),
        VWAPStrategy(vwap_window=300),
        SpreadArbitrageStrategy(spread_threshold=0.0003)
    ]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"戦略: {strategy.name}")
        
        # エンジン作成
        engine = BacktestEngine(
            initial_capital=10_000_000,  # 1000万円
            commission_rate=0.0003,
            slippage_rate=0.0001
        )
        
        # 戦略セット
        engine.set_strategy(strategy)
        
        # バックテスト実行
        engine.run(tick_data)
        
        # 結果保存
        results[strategy.name] = {
            'trades': [vars(trade) for trade in engine.trades],
            'equity_curve': engine.equity_curve,
            'initial_capital': engine.initial_capital,
            'metrics': engine.get_performance_metrics()
        }
        
        # メトリクス表示
        metrics = results[strategy.name]['metrics']
        logger.info(f"  総リターン: {metrics.get('total_return', 0):.2%}")
        logger.info(f"  シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  最大ドローダウン: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  勝率: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  総取引数: {metrics.get('total_trades', 0)}")
    
    # 5. 結果分析・可視化
    logger.info("5. 結果分析とレポート生成")
    
    # 最良の戦略を選択
    if results:
        best_strategy = max(results.items(), 
                           key=lambda x: x[1]['metrics'].get('sharpe_ratio', 0))
        
        logger.info(f"最良戦略: {best_strategy[0]}")
        
        # アナライザー作成
        analyzer = BacktestAnalyzer(best_strategy[1])
        
        # レポート生成
        report_path = f"reports/backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        os.makedirs("reports", exist_ok=True)
        analyzer.generate_report(report_path)
        
        logger.info(f"レポート生成完了: {report_path}")
        
        # プロット表示（オプション）
        SHOW_PLOTS = os.getenv('SHOW_PLOTS', 'false').lower() == 'true'
        if SHOW_PLOTS and ohlc_data is not None and not ohlc_data.empty:
            fig = analyzer.plot_equity_curve()
            fig.show()
    
    # クリーンアップ
    if USE_BLOOMBERG and 'fetcher' in locals():
        fetcher.disconnect()
    if storage:
        storage.close()


def generate_demo_tick_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """デモ用のティックデータを生成"""
    logger.info("デモデータを生成します")
    
    # 時間範囲
    date_range = pd.date_range(start=start_date, end=end_date, freq='S')
    
    # 基準価格
    base_price = 2000.0
    
    tick_data = []
    current_bid = base_price - 0.5
    current_ask = base_price + 0.5
    
    for timestamp in date_range:
        # ランダムウォーク
        price_change = np.random.normal(0, 0.1)
        base_price += price_change
        
        # BID/ASK更新
        if np.random.random() < 0.3:
            current_bid = base_price - np.random.uniform(0.3, 0.7)
            tick_data.append({
                'time': timestamp,
                'type': 'BID',
                'value': current_bid,
                'size': np.random.randint(100, 1000) * 100,
                'exchange': 'TSE',
                'condition': ''
            })
        
        if np.random.random() < 0.3:
            current_ask = base_price + np.random.uniform(0.3, 0.7)
            tick_data.append({
                'time': timestamp,
                'type': 'ASK',
                'value': current_ask,
                'size': np.random.randint(100, 1000) * 100,
                'exchange': 'TSE',
                'condition': ''
            })
        
        # TRADE
        if np.random.random() < 0.5:
            trade_price = np.random.uniform(current_bid, current_ask)
            tick_data.append({
                'time': timestamp,
                'type': 'TRADE',
                'value': trade_price,
                'size': np.random.randint(1, 50) * 100,
                'exchange': 'TSE',
                'condition': ''
            })
    
    df = pd.DataFrame(tick_data)
    df['symbol'] = symbol
    
    return df


if __name__ == "__main__":
    setup_logging()
    
    try:
        run_backtest_example()
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)