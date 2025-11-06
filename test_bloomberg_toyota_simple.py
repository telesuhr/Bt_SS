"""
Bloomberg APIを使用してトヨタのティックデータを取得しバックテストを実行
2024/11/7のデータを使用（簡易版）
"""
import os
import sys
from datetime import datetime

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 簡易ログ設定
class SimpleLogger:
    @staticmethod
    def info(msg):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    @staticmethod
    def error(msg):
        print(f"[ERROR] {datetime.now().strftime('%H:%M:%S')} - {msg}")
    
    @staticmethod
    def warning(msg):
        print(f"[WARNING] {datetime.now().strftime('%H:%M:%S')} - {msg}")

logger = SimpleLogger()


def test_bloomberg_connection():
    """Bloomberg API接続テスト"""
    try:
        import blpapi
        logger.info("blpapi module imported successfully")
        
        from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
        fetcher = TickDataFetcher()
        
        if fetcher.connect():
            logger.info("Bloomberg API connection successful")
            fetcher.disconnect()
            return True
        else:
            logger.error("Bloomberg API connection failed")
            return False
            
    except ImportError as e:
        logger.error(f"blpapi import error: {e}")
        logger.info("Please install Bloomberg API Python library manually")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def run_toyota_backtest():
    """トヨタのバックテストを実行（簡易版）"""
    try:
        from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
        from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
        from src.backtest.engine import BacktestEngine
        from src.strategies.base_strategy import TickMomentumStrategy, VWAPStrategy
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return
    
    # パラメータ設定
    SYMBOL = "7203 JT Equity"  # トヨタ自動車
    # 今日の日付を使用（2025年11月7日）
    today = datetime.now().date()
    START_DATE = datetime(today.year, today.month, today.day, 9, 0, 0)
    END_DATE = datetime(today.year, today.month, today.day, 15, 0, 0)
    
    logger.info("=== Toyota Tick Data Backtest ===")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Period: {START_DATE} - {END_DATE}")
    
    # Bloomberg APIからデータ取得
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        logger.error("Failed to connect Bloomberg API")
        logger.info("Please check Bloomberg Terminal is running")
        return
    
    try:
        logger.info("Fetching tick data...")
        tick_data = fetcher.fetch_tick_data(SYMBOL, START_DATE, END_DATE)
        
        if tick_data.empty:
            logger.error("No tick data fetched")
            logger.info("Possible reasons:")
            logger.info("1. Bloomberg Terminal not logged in")
            logger.info("2. API service not running")
            logger.info("3. Wrong symbol code (check 7203 JT Equity)")
            logger.info("4. Market closed at specified time")
            return
        
        logger.info(f"Fetched: {len(tick_data)} ticks")
        
        # データの概要を表示
        logger.info("\n=== Data Summary ===")
        logger.info(f"Start: {tick_data['time'].min()}")
        logger.info(f"End: {tick_data['time'].max()}")
        logger.info("Tick types:")
        print(tick_data['type'].value_counts())
        
        # バックテスト実行
        logger.info("\n=== Running Backtest ===")
        
        # シンプルな戦略でテスト
        strategy = TickMomentumStrategy(lookback_ticks=50)
        logger.info(f"Strategy: {strategy.name}")
        
        engine = BacktestEngine(
            initial_capital=10_000_000,  # 1000万円
            commission_rate=0.0003,      # 0.03%
            slippage_rate=0.0001         # 0.01%
        )
        
        engine.set_strategy(strategy)
        engine.run(tick_data)
        
        metrics = engine.get_performance_metrics()
        
        logger.info("\n=== Results ===")
        logger.info(f"Total Return: {metrics.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Final Equity: {metrics.get('final_equity', 0):,.0f} yen")
        
        # 取引詳細（最初の5件）
        if engine.trades:
            logger.info("\n=== First 5 Trades ===")
            for i, trade in enumerate(engine.trades[:5]):
                logger.info(f"Trade {i+1}: {trade.side.value} {trade.quantity} shares "
                          f"@ {trade.price:.1f} yen at {trade.timestamp}")
        
        # エクイティカーブをCSVで保存
        if engine.equity_curve:
            import pandas as pd
            equity_df = pd.DataFrame(engine.equity_curve)
            csv_path = f"reports/toyota_equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.makedirs("reports", exist_ok=True)
            equity_df.to_csv(csv_path, index=False)
            logger.info(f"\nEquity curve saved: {csv_path}")
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()
        logger.info("\n=== Completed ===")


if __name__ == "__main__":
    logger.info("Testing Bloomberg API connection...")
    
    if test_bloomberg_connection():
        logger.info("\nStarting backtest...")
        run_toyota_backtest()
    else:
        logger.error("\nCannot connect to Bloomberg API")
        logger.info("\nSolutions:")
        logger.info("1. Check Bloomberg Terminal is running")
        logger.info("2. Check logged into Bloomberg Terminal")
        logger.info("3. Install blpapi:")
        logger.info("   pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi")