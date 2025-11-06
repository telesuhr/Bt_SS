"""
Bloomberg APIを使用してトヨタの分足データを取得してテスト
"""
import os
import sys
from datetime import datetime, timedelta

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

logger = SimpleLogger()


def test_intraday_bars():
    """分足データ取得テスト"""
    try:
        from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
        
        # パラメータ設定
        SYMBOL = "7203 JT Equity"  # トヨタ自動車
        
        # 過去5営業日のデータを試す
        END_DATE = datetime.now()
        START_DATE = END_DATE - timedelta(days=7)
        
        logger.info(f"=== Fetching Intraday Bars for {SYMBOL} ===")
        logger.info(f"Period: {START_DATE} - {END_DATE}")
        
        fetcher = TickDataFetcher()
        
        if not fetcher.connect():
            logger.error("Failed to connect to Bloomberg API")
            return
        
        try:
            # 1分足データを取得
            logger.info("Fetching 1-minute bars...")
            bar_data = fetcher.fetch_intraday_bars(
                SYMBOL, 
                START_DATE, 
                END_DATE,
                interval=1  # 1分足
            )
            
            if bar_data.empty:
                logger.error("No bar data fetched")
                
                # 代替の銘柄コードを試す
                alternative_symbols = [
                    "7203 JP Equity",  # JP
                    "7203 JT Equity",  # JT (Tokyo)
                    "7203.T",          # .T
                    "7203 Equity"      # Equity only
                ]
                
                logger.info("Trying alternative symbols...")
                for alt_symbol in alternative_symbols:
                    logger.info(f"Trying: {alt_symbol}")
                    bar_data = fetcher.fetch_intraday_bars(
                        alt_symbol, 
                        START_DATE, 
                        END_DATE,
                        interval=1
                    )
                    if not bar_data.empty:
                        logger.info(f"Success with symbol: {alt_symbol}")
                        SYMBOL = alt_symbol
                        break
            
            if not bar_data.empty:
                logger.info(f"Fetched: {len(bar_data)} bars")
                logger.info("\n=== Data Sample ===")
                print(bar_data.head(10))
                
                logger.info("\n=== Data Summary ===")
                logger.info(f"Start: {bar_data['time'].min()}")
                logger.info(f"End: {bar_data['time'].max()}")
                logger.info(f"Open: Min={bar_data['open'].min():.2f}, Max={bar_data['open'].max():.2f}")
                logger.info(f"Volume: Total={bar_data['volume'].sum():,}")
                
                # CSVに保存
                csv_path = f"reports/toyota_bars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.makedirs("reports", exist_ok=True)
                bar_data.to_csv(csv_path, index=False)
                logger.info(f"\nData saved to: {csv_path}")
            else:
                logger.error("Could not fetch data with any symbol variation")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            fetcher.disconnect()
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Please install required packages")


if __name__ == "__main__":
    test_intraday_bars()