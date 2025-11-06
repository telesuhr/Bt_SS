"""
TA-Lib不要の軽量版テクニカル指標計算ライブラリ
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class TechnicalIndicatorsLite:
    """TA-Libを使わないテクニカル指標計算クラス"""
    
    @staticmethod
    def tick_to_ohlc(tick_df: pd.DataFrame, 
                     interval: str = '1Min',
                     price_col: str = 'value') -> pd.DataFrame:
        """ティックデータをOHLCデータに変換"""
        tick_df = tick_df.copy()
        tick_df.set_index('time', inplace=True)
        
        trade_data = tick_df[tick_df['type'] == 'TRADE'].copy()
        
        ohlc = trade_data.groupby(pd.Grouper(freq=interval)).agg({
            price_col: ['first', 'max', 'min', 'last'],
            'size': 'sum'
        })
        
        ohlc.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlc = ohlc.dropna()
        
        return ohlc
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """単純移動平均"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移動平均"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(data: pd.Series, 
                       period: int = 20, 
                       std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド"""
        middle = TechnicalIndicatorsLite.sma(data, period)
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def macd(data: pd.Series,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD"""
        ema_fast = TechnicalIndicatorsLite.ema(data, fast_period)
        ema_slow = TechnicalIndicatorsLite.ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicatorsLite.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """RSI"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stochastic(high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   fast_k: int = 5,
                   slow_k: int = 3,
                   slow_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス"""
        lowest_low = low.rolling(window=fast_k).min()
        highest_high = high.rolling(window=fast_k).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        slow_k = k_percent.rolling(window=slow_k).mean()
        slow_d = slow_k.rolling(window=slow_d).mean()
        
        return slow_k, slow_d
    
    @staticmethod
    def vwap(price: pd.Series, volume: pd.Series, window: Optional[int] = None) -> pd.Series:
        """VWAP（出来高加重平均価格）"""
        pv = price * volume
        
        if window:
            cumulative_pv = pv.rolling(window=window).sum()
            cumulative_volume = volume.rolling(window=window).sum()
        else:
            cumulative_pv = pv.cumsum()
            cumulative_volume = volume.cumsum()
        
        return cumulative_pv / cumulative_volume
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """OBV（オンバランスボリューム）"""
        direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
        return (direction * volume).cumsum()
    
    @staticmethod
    def atr(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            period: int = 14) -> pd.Series:
        """ATR（アベレージ・トゥルー・レンジ）"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_spread(tick_df: pd.DataFrame) -> pd.Series:
        """BID-ASKスプレッドを計算"""
        bid_data = tick_df[tick_df['type'] == 'BID'][['time', 'value']].set_index('time')
        ask_data = tick_df[tick_df['type'] == 'ASK'][['time', 'value']].set_index('time')
        
        spread_df = pd.DataFrame({
            'bid': bid_data['value'],
            'ask': ask_data['value']
        })
        
        spread_df = spread_df.fillna(method='ffill')
        spread = spread_df['ask'] - spread_df['bid']
        
        return spread
    
    @staticmethod
    def tick_intensity(tick_df: pd.DataFrame, window: str = '1Min') -> pd.Series:
        """ティック強度（約定頻度）を計算"""
        tick_df_copy = tick_df.copy()
        tick_df_copy.set_index('time', inplace=True)
        tick_counts = tick_df_copy[tick_df_copy['type'] == 'TRADE'].resample(window).size()
        return tick_counts
    
    @staticmethod
    def order_flow_imbalance(tick_df: pd.DataFrame, window: str = '1Min') -> pd.Series:
        """注文フローの偏りを計算"""
        tick_df_copy = tick_df.copy()
        tick_df_copy.set_index('time', inplace=True)
        trade_data = tick_df_copy[tick_df_copy['type'] == 'TRADE'].copy()
        
        trade_data['price_change'] = trade_data['value'].diff()
        trade_data['buy_volume'] = trade_data.apply(
            lambda x: x['size'] if x['price_change'] > 0 else 0, axis=1
        )
        trade_data['sell_volume'] = trade_data.apply(
            lambda x: x['size'] if x['price_change'] < 0 else 0, axis=1
        )
        
        buy_volume = trade_data['buy_volume'].resample(window).sum()
        sell_volume = trade_data['sell_volume'].resample(window).sum()
        
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / total_volume.replace(0, 1)
        
        return imbalance