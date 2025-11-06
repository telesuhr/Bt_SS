"""
Bloomberg APIを使用してトヨタのティックデータを取得しバックテストを実行
2024/11/7のデータを使用
"""
import os
import sys
from datetime import datetime
from loguru import logger

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ログ設定
logger.remove()
logger.add(sys.stdout, level="INFO")

def test_bloomberg_connection():
    """Bloomberg API接続テスト"""
    try:
        import blpapi
        logger.info("blpapi module imported successfully")
        
        from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
        fetcher = TickDataFetcher()
        
        if fetcher.connect():
            logger.info("✓ Bloomberg API connection successful")
            fetcher.disconnect()
            return True
        else:
            logger.error("✗ Bloomberg API connection failed")
            return False
            
    except ImportError as e:
        logger.error(f"✗ blpapi import error: {e}")
        logger.info("Please install Bloomberg API Python library manually")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        return False


def run_toyota_backtest():
    """トヨタのバックテストを実行"""
    from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
    from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
    from src.backtest.engine import BacktestEngine
    from src.strategies.base_strategy import TickMomentumStrategy, VWAPStrategy
    from src.analysis.analyzer import BacktestAnalyzer
    
    # パラメータ設定
    SYMBOL = "7203 JT Equity"  # トヨタ自動車
    START_DATE = datetime(2024, 11, 7, 9, 0, 0)
    END_DATE = datetime(2024, 11, 7, 15, 0, 0)
    
    logger.info(f"=== トヨタ ティックデータバックテスト ===")
    logger.info(f"銘柄: {SYMBOL}")
    logger.info(f"期間: {START_DATE} ～ {END_DATE}")
    
    # Bloomberg APIからデータ取得
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        logger.error("Bloomberg API接続に失敗しました")
        logger.info("Bloomberg Terminalが起動していることを確認してください")
        return
    
    try:
        logger.info("ティックデータを取得中...")
        tick_data = fetcher.fetch_tick_data(SYMBOL, START_DATE, END_DATE)
        
        if tick_data.empty:
            logger.error("ティックデータが取得できませんでした")
            logger.info("考えられる原因:")
            logger.info("1. Bloomberg Terminalにログインしていない")
            logger.info("2. APIサービスが起動していない")
            logger.info("3. 銘柄コードが正しくない（7203 JT Equityを確認）")
            logger.info("4. 指定日時の市場が開いていない")
            return
        
        logger.info(f"✓ 取得成功: {len(tick_data)}ティック")
        
        # データの概要を表示
        logger.info("\n=== データ概要 ===")
        logger.info(f"開始時刻: {tick_data['time'].min()}")
        logger.info(f"終了時刻: {tick_data['time'].max()}")
        logger.info(f"ティックタイプ別件数:")
        logger.info(tick_data['type'].value_counts().to_string())
        
        # テクニカル指標計算
        logger.info("\n=== テクニカル指標計算 ===")
        indicators = TechnicalIndicators()
        
        # 1分足データ生成
        ohlc_data = indicators.tick_to_ohlc(tick_data, interval='1Min')
        logger.info(f"1分足データ: {len(ohlc_data)}本")
        
        if len(ohlc_data) < 20:
            logger.warning("データが少なすぎます。より長い期間のデータを取得してください")
            return
        
        # バックテスト実行
        logger.info("\n=== バックテスト実行 ===")
        
        # シンプルな戦略でテスト
        strategies = [
            TickMomentumStrategy(lookback_ticks=50),
            VWAPStrategy(vwap_window=100)
        ]
        
        best_result = None
        best_sharpe = -999
        
        for strategy in strategies:
            logger.info(f"\n戦略: {strategy.name}")
            
            engine = BacktestEngine(
                initial_capital=10_000_000,  # 1000万円
                commission_rate=0.0003,      # 0.03%
                slippage_rate=0.0001         # 0.01%
            )
            
            engine.set_strategy(strategy)
            engine.run(tick_data)
            
            metrics = engine.get_performance_metrics()
            
            logger.info(f"総リターン: {metrics.get('total_return', 0):.2%}")
            logger.info(f"シャープレシオ: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"最大ドローダウン: {metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"総取引数: {metrics.get('total_trades', 0)}")
            
            if metrics.get('sharpe_ratio', -999) > best_sharpe:
                best_sharpe = metrics.get('sharpe_ratio', -999)
                best_result = {
                    'trades': [vars(trade) for trade in engine.trades],
                    'equity_curve': engine.equity_curve,
                    'initial_capital': engine.initial_capital,
                    'metrics': metrics,
                    'strategy_name': strategy.name
                }
        
        # レポート生成
        if best_result:
            logger.info(f"\n=== 最良戦略: {best_result['strategy_name']} ===")
            
            analyzer = BacktestAnalyzer(best_result)
            
            report_path = f"reports/toyota_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            os.makedirs("reports", exist_ok=True)
            analyzer.generate_report(report_path)
            
            logger.info(f"✓ レポート生成完了: {report_path}")
            
            # 取引詳細
            if best_result['trades']:
                logger.info(f"\n=== 取引詳細（最初の5件） ===")
                for i, trade in enumerate(best_result['trades'][:5]):
                    logger.info(f"取引{i+1}: {trade['side']} {trade['quantity']}株 "
                              f"@{trade['price']:.1f}円 "
                              f"時刻: {trade['timestamp']}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
    
    finally:
        fetcher.disconnect()
        logger.info("\n=== 完了 ===")


if __name__ == "__main__":
    logger.info("Bloomberg API接続テスト...")
    
    if test_bloomberg_connection():
        logger.info("\nバックテストを開始します...")
        run_toyota_backtest()
    else:
        logger.error("\nBloomberg APIに接続できません")
        logger.info("\n対処法:")
        logger.info("1. Bloomberg Terminalが起動していることを確認")
        logger.info("2. Bloomberg Terminalにログイン済みであることを確認")
        logger.info("3. 以下のコマンドでblpapiをインストール:")
        logger.info("   pip install --index-url=https://bcms.bloomberg.com/pip/simple blpapi")