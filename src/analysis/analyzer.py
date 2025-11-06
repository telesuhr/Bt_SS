import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger


class BacktestAnalyzer:
    """バックテスト結果分析クラス"""
    
    def __init__(self, backtest_result: Dict):
        self.trades = pd.DataFrame(backtest_result.get('trades', []))
        self.equity_curve = pd.DataFrame(backtest_result.get('equity_curve', []))
        self.initial_capital = backtest_result.get('initial_capital', 1_000_000)
        
        if not self.equity_curve.empty:
            self.equity_curve.set_index('timestamp', inplace=True)
        
        if not self.trades.empty:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
    
    def calculate_metrics(self) -> Dict:
        """詳細なパフォーマンス指標を計算"""
        if self.equity_curve.empty:
            return {}
        
        # 基本メトリクス
        final_equity = self.equity_curve['equity'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 日次リターン
        daily_equity = self.equity_curve['equity'].resample('D').last()
        daily_returns = daily_equity.pct_change().dropna()
        
        # 年率換算リターン
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # ボラティリティ
        volatility = daily_returns.std() * np.sqrt(252)
        
        # シャープレシオ
        sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0
        
        # ソルティノレシオ
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return / downside_volatility) if downside_volatility > 0 else 0
        
        # 最大ドローダウン
        rolling_max = self.equity_curve['equity'].expanding().max()
        drawdown = (self.equity_curve['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # ドローダウン期間
        drawdown_periods = self._calculate_drawdown_periods(drawdown)
        max_drawdown_duration = max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0
        
        # 取引統計
        trade_stats = self._calculate_trade_statistics()
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'final_equity': final_equity,
            **trade_stats
        }
        
        return metrics
    
    def _calculate_trade_statistics(self) -> Dict:
        """取引統計を計算"""
        if self.trades.empty:
            return {}
        
        # 売買ペアで利益を計算
        buy_trades = self.trades[self.trades['side'] == 'BUY'].copy()
        sell_trades = self.trades[self.trades['side'] == 'SELL'].copy()
        
        profits = []
        for _, sell in sell_trades.iterrows():
            # 対応する買い注文を探す
            matching_buys = buy_trades[
                (buy_trades['symbol'] == sell['symbol']) & 
                (buy_trades['timestamp'] < sell['timestamp'])
            ]
            
            if not matching_buys.empty:
                buy = matching_buys.iloc[-1]
                profit = (sell['price'] - buy['price']) * sell['quantity'] - \
                        (sell['commission'] + buy['commission'])
                profits.append(profit)
        
        if not profits:
            return {
                'total_trades': len(self.trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'expectancy': 0
            }
        
        profits = np.array(profits)
        winning_trades = profits[profits > 0]
        losing_trades = profits[profits < 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(profits) if profits.size > 0 else 0,
            'avg_win': winning_trades.mean() if winning_trades.size > 0 else 0,
            'avg_loss': losing_trades.mean() if losing_trades.size > 0 else 0,
            'profit_factor': abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() < 0 else np.inf,
            'expectancy': profits.mean() if profits.size > 0 else 0
        }
    
    def _calculate_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        """ドローダウン期間を計算"""
        periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_date = date
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                if start_date:
                    periods.append({
                        'start': start_date,
                        'end': date,
                        'duration': (date - start_date).days
                    })
        
        return periods
    
    def plot_equity_curve(self) -> go.Figure:
        """エクイティカーブをプロット"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )
        
        # エクイティカーブ
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 初期資本ライン
        fig.add_hline(
            y=self.initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital",
            row=1, col=1
        )
        
        # ドローダウン
        rolling_max = self.equity_curve['equity'].expanding().max()
        drawdown = (self.equity_curve['equity'] - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=1),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Backtest Performance',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        
        return fig
    
    def plot_trades(self, price_data: pd.DataFrame) -> go.Figure:
        """取引をプロット"""
        fig = go.Figure()
        
        # 価格チャート
        fig.add_trace(
            go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name='Price'
            )
        )
        
        # 買い注文
        buy_trades = self.trades[self.trades['side'] == 'BUY']
        fig.add_trace(
            go.Scatter(
                x=buy_trades['timestamp'],
                y=buy_trades['price'],
                mode='markers',
                name='Buy',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='triangle-up'
                )
            )
        )
        
        # 売り注文
        sell_trades = self.trades[self.trades['side'] == 'SELL']
        fig.add_trace(
            go.Scatter(
                x=sell_trades['timestamp'],
                y=sell_trades['price'],
                mode='markers',
                name='Sell',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='triangle-down'
                )
            )
        )
        
        fig.update_layout(
            title='Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600
        )
        
        return fig
    
    def plot_return_distribution(self) -> go.Figure:
        """リターン分布をプロット"""
        daily_returns = self.equity_curve['equity'].pct_change().dropna()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Return Distribution', 'Q-Q Plot')
        )
        
        # ヒストグラム
        fig.add_trace(
            go.Histogram(
                x=daily_returns,
                nbinsx=50,
                name='Returns',
                histnorm='probability'
            ),
            row=1, col=1
        )
        
        # 正規分布フィット
        mean = daily_returns.mean()
        std = daily_returns.std()
        x_range = np.linspace(daily_returns.min(), daily_returns.max(), 100)
        normal_dist = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist * (daily_returns.max() - daily_returns.min()) / 50,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Q-Qプロット
        sorted_returns = np.sort(daily_returns)
        norm_quantiles = np.linspace(0.01, 0.99, len(sorted_returns))
        theoretical_quantiles = np.percentile(
            np.random.normal(mean, std, 10000), 
            norm_quantiles * 100
        )
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_returns,
                mode='markers',
                name='Q-Q'
            ),
            row=1, col=2
        )
        
        # 45度線
        min_val = min(theoretical_quantiles.min(), sorted_returns.min())
        max_val = max(theoretical_quantiles.max(), sorted_returns.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='45° line',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=True)
        fig.update_xaxes(title_text="Return", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        return fig
    
    def generate_report(self, output_path: str = "backtest_report.html"):
        """総合レポートを生成"""
        metrics = self.calculate_metrics()
        
        # HTML テンプレート
        html_template = f"""
        <html>
        <head>
            <title>Backtest Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .section {{ margin: 30px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Backtest Performance Report</h1>
            
            <div class="section">
                <h2>Key Performance Metrics</h2>
                <div class="metric">
                    <div class="metric-value">{metrics.get('total_return', 0):.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('max_drawdown', 0):.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('win_rate', 0):.2%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Detailed Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Annual Return</td><td>{metrics.get('annual_return', 0):.2%}</td></tr>
                    <tr><td>Volatility</td><td>{metrics.get('volatility', 0):.2%}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{metrics.get('sortino_ratio', 0):.2f}</td></tr>
                    <tr><td>Total Trades</td><td>{metrics.get('total_trades', 0)}</td></tr>
                    <tr><td>Profit Factor</td><td>{metrics.get('profit_factor', 0):.2f}</td></tr>
                    <tr><td>Expectancy</td><td>¥{metrics.get('expectancy', 0):,.0f}</td></tr>
                </table>
            </div>
        """
        
        # プロット追加
        if not self.equity_curve.empty:
            equity_fig = self.plot_equity_curve()
            html_template += f"""
            <div class="section">
                <div id="equity_plot"></div>
                <script>
                    Plotly.newPlot('equity_plot', {equity_fig.to_json()});
                </script>
            </div>
            """
        
        html_template += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"Report generated: {output_path}")