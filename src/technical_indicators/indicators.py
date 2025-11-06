import pandas as pd
import numpy as np
from typing import Optional, Tuple
import talib
from loguru import logger


class TechnicalIndicators:
    """ティックデータ用テクニカル指標計算クラス"""
    
    @staticmethod
    def tick_to_ohlc(tick_df: pd.DataFrame, 
                     interval: str = '1Min',
                     price_col: str = 'value') -> pd.DataFrame:
        """
        ティックデータをOHLCデータに変換
        
        Parameters:
        -----------
        tick_df : pd.DataFrame
            ティックデータ（time, value, size列を含む）
        interval : str
            時間間隔（'1Min', '5Min', '1H'等）
        price_col : str
            価格列名
        """
        tick_df = tick_df.copy()
        tick_df.set_index('time', inplace=True)
        
        # トレードデータのみ抽出
        trade_data = tick_df[tick_df['type'] == 'TRADE'].copy()
        
        ohlc_dict = {
            price_col: 'ohlc',
            'size': 'sum'
        }
        
        ohlc = trade_data.resample(interval).agg(ohlc_dict)
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        
        # NaNを前の値で埋める
        ohlc.fillna(method='ffill', inplace=True)
        
        return ohlc
    
    @staticmethod
    def calculate_spread(tick_df: pd.DataFrame) -> pd.Series:
        """BID-ASKスプレッドを計算"""
        bid_data = tick_df[tick_df['type'] == 'BID']['value']
        ask_data = tick_df[tick_df['type'] == 'ASK']['value']
        
        # 時間でアラインメント
        spread_df = pd.DataFrame({
            'bid': bid_data,
            'ask': ask_data
        })
        
        spread_df.fillna(method='ffill', inplace=True)
        spread = spread_df['ask'] - spread_df['bid']
        
        return spread
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """単純移動平均"""
        return talib.SMA(data, timeperiod=period)
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移動平均"""
        return talib.EMA(data, timeperiod=period)
    
    @staticmethod
    def bollinger_bands(data: pd.Series, 
                       period: int = 20, 
                       std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        upper, middle, lower = talib.BBANDS(
            data, 
            timeperiod=period,
            nbdevup=std_dev,
            nbdevdn=std_dev
        )
        return upper, middle, lower
    
    @staticmethod
    def macd(data: pd.Series,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        macd, signal, hist = talib.MACD(
            data,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        return macd, signal, hist
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """RSI"""
        return talib.RSI(data, timeperiod=period)
    
    @staticmethod
    def stochastic(high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   fast_k: int = 5,
                   slow_k: int = 3,
                   slow_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス"""
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=fast_k,
            slowk_period=slow_k,
            slowd_period=slow_d
        )
        return slowk, slowd
    
    @staticmethod
    def vwap(price: pd.Series, volume: pd.Series) -> pd.Series:
        """VWAP（出来高加重平均価格）"""
        typical_price = price
        cumulative_pv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        return cumulative_pv / cumulative_volume
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV（オンバランスボリューム）"""
        return talib.OBV(close, volume)
    
    @staticmethod
    def atr(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            period: int = 14) -> pd.Series:
        """ATR（アベレージ・トゥルー・レンジ）"""
        return talib.ATR(high, low, close, timeperiod=period)
    
    @staticmethod
    def tick_intensity(tick_df: pd.DataFrame, window: str = '1Min') -> pd.Series:
        """
        ティック強度（約定頻度）を計算
        
        高頻度取引の活発度を測る指標
        """
        tick_counts = tick_df[tick_df['type'] == 'TRADE'].resample(window).size()
        return tick_counts
    
    @staticmethod
    def order_flow_imbalance(tick_df: pd.DataFrame, window: str = '1Min') -> pd.Series:
        """
        注文フローの偏りを計算
        
        買い圧力と売り圧力のバランスを測定
        """
        trade_data = tick_df[tick_df['type'] == 'TRADE'].copy()
        
        # 価格の変化方向で買い/売りを推定
        trade_data['price_change'] = trade_data['value'].diff()
        trade_data['buy_volume'] = trade_data.apply(
            lambda x: x['size'] if x['price_change'] > 0 else 0, axis=1
        )
        trade_data['sell_volume'] = trade_data.apply(
            lambda x: x['size'] if x['price_change'] < 0 else 0, axis=1
        )
        
        # リサンプリング
        buy_volume = trade_data['buy_volume'].resample(window).sum()
        sell_volume = trade_data['sell_volume'].resample(window).sum()
        
        # インバランス計算
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / total_volume.replace(0, 1)
        
        return imbalance
    
    @staticmethod
    def microstructure_features(tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        マイクロストラクチャー特徴量を計算
        
        高頻度取引向けの詳細な市場構造分析
        """
        features = pd.DataFrame(index=tick_df.index)
        
        # スプレッド
        features['spread'] = TechnicalIndicators.calculate_spread(tick_df)
        
        # 相対スプレッド
        mid_price = (tick_df[tick_df['type'] == 'BID']['value'] + 
                    tick_df[tick_df['type'] == 'ASK']['value']) / 2
        features['relative_spread'] = features['spread'] / mid_price
        
        # 約定サイズの移動平均
        trade_sizes = tick_df[tick_df['type'] == 'TRADE']['size']
        features['avg_trade_size'] = trade_sizes.rolling(window=100).mean()
        
        # ティック方向指標
        trade_prices = tick_df[tick_df['type'] == 'TRADE']['value']
        features['tick_direction'] = np.sign(trade_prices.diff())
        
        return features