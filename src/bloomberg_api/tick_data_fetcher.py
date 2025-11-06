try:
    import blpapi
except ImportError:
    logger.error("blpapi not installed. Please install it manually.")
    raise
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger


class TickDataFetcher:
    """Bloomberg APIを使用してティックデータを取得するクラス"""
    
    def __init__(self, host: str = "localhost", port: int = 8194):
        self.host = host
        self.port = port
        self.session = None
        self.service = None
        
    def connect(self) -> bool:
        """Bloomberg APIに接続"""
        try:
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost(self.host)
            sessionOptions.setServerPort(self.port)
            
            self.session = blpapi.Session(sessionOptions)
            
            if not self.session.start():
                logger.error("Failed to start Bloomberg session")
                return False
                
            if not self.session.openService("//blp/refdata"):
                logger.error("Failed to open Bloomberg service")
                return False
                
            self.service = self.session.getService("//blp/refdata")
            logger.info("Successfully connected to Bloomberg API")
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def fetch_tick_data(self, 
                       ticker: str, 
                       start_date: datetime, 
                       end_date: datetime,
                       fields: List[str] = None) -> pd.DataFrame:
        """
        ティックデータを取得
        
        Parameters:
        -----------
        ticker : str
            銘柄コード（例: "7203 JT Equity" for トヨタ）
        start_date : datetime
            開始日時
        end_date : datetime
            終了日時
        fields : List[str]
            取得フィールド（デフォルト: TRADE, BID, ASK）
        """
        if fields is None:
            fields = ["TRADE", "BID", "ASK", "VOLUME"]
        
        try:
            request = self.service.createRequest("IntradayTickRequest")
            
            request.set("security", ticker)
            request.append("eventTypes", "TRADE")
            request.append("eventTypes", "BID")
            request.append("eventTypes", "ASK")
            
            request.set("startDateTime", start_date)
            request.set("endDateTime", end_date)
            request.set("includeConditionCodes", True)
            request.set("includeExchangeCodes", True)
            
            logger.info(f"Fetching tick data for {ticker} from {start_date} to {end_date}")
            
            self.session.sendRequest(request)
            
            tick_data = []
            
            while True:
                ev = self.session.nextEvent(500)
                
                for msg in ev:
                    if msg.hasElement("tickData"):
                        tick_array = msg.getElement("tickData")
                        tick_array_dict = tick_array.getElement("tickData")
                        
                        for i in range(tick_array_dict.numValues()):
                            tick = tick_array_dict.getValueAsElement(i)
                            
                            tick_dict = {
                                "time": tick.getElementAsDatetime("time"),
                                "type": tick.getElementAsString("type"),
                                "value": tick.getElementAsFloat("value"),
                                "size": tick.getElementAsInteger("size") if tick.hasElement("size") else 0,
                                "condition": tick.getElementAsString("conditionCode") if tick.hasElement("conditionCode") else "",
                                "exchange": tick.getElementAsString("exchangeCode") if tick.hasElement("exchangeCode") else ""
                            }
                            
                            tick_data.append(tick_dict)
                
                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            
            df = pd.DataFrame(tick_data)
            
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
                df = df.reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} ticks for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching tick data: {e}")
            return pd.DataFrame()
    
    def fetch_intraday_bars(self,
                           ticker: str,
                           start_date: datetime,
                           end_date: datetime,
                           interval: int = 1) -> pd.DataFrame:
        """
        分足データを取得
        
        Parameters:
        -----------
        ticker : str
            銘柄コード
        start_date : datetime
            開始日時
        end_date : datetime
            終了日時
        interval : int
            バーの間隔（分）
        """
        try:
            request = self.service.createRequest("IntradayBarRequest")
            
            request.set("security", ticker)
            request.set("eventType", "TRADE")
            request.set("startDateTime", start_date)
            request.set("endDateTime", end_date)
            request.set("interval", interval)
            
            logger.info(f"Fetching {interval}-minute bars for {ticker}")
            
            self.session.sendRequest(request)
            
            bars = []
            
            while True:
                ev = self.session.nextEvent(500)
                
                for msg in ev:
                    if msg.hasElement("barData"):
                        bar_array = msg.getElement("barData")
                        bar_tick_array = bar_array.getElement("barTickData")
                        
                        for i in range(bar_tick_array.numValues()):
                            bar = bar_tick_array.getValueAsElement(i)
                            
                            bar_dict = {
                                "time": bar.getElementAsDatetime("time"),
                                "open": bar.getElementAsFloat("open"),
                                "high": bar.getElementAsFloat("high"),
                                "low": bar.getElementAsFloat("low"),
                                "close": bar.getElementAsFloat("close"),
                                "volume": bar.getElementAsInteger("volume"),
                                "numEvents": bar.getElementAsInteger("numEvents")
                            }
                            
                            bars.append(bar_dict)
                
                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            
            df = pd.DataFrame(bars)
            
            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
                df = df.reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} bars for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching bar data: {e}")
            return pd.DataFrame()
    
    def disconnect(self):
        """Bloomberg APIとの接続を切断"""
        if self.session:
            self.session.stop()
            logger.info("Disconnected from Bloomberg API")