"""
トヨタのバックテスト結果をHTMLチャートで可視化
"""
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import json

# モジュールパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bloomberg_api.tick_data_fetcher import TickDataFetcher
from src.technical_indicators.indicators_lite import TechnicalIndicatorsLite as TechnicalIndicators
from src.backtest.engine import BacktestEngine, OrderSide
from backtest_toyota_bars import SimpleMAStrategy, convert_bars_to_ticks


def create_interactive_chart():
    """インタラクティブなHTMLチャートを作成"""
    print("[INFO] トヨタ バックテスト結果の可視化（HTML版）")
    
    # Bloomberg APIから分足データ取得
    SYMBOL = "7203 JT Equity"
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=7)
    
    fetcher = TickDataFetcher()
    
    if not fetcher.connect():
        print("[ERROR] Bloomberg API接続失敗")
        return
    
    try:
        # データ取得
        print("[INFO] データ取得中...")
        bar_data = fetcher.fetch_intraday_bars(SYMBOL, START_DATE, END_DATE, interval=1)
        
        if bar_data.empty:
            print("[ERROR] データが取得できませんでした")
            return
        
        print(f"[INFO] {len(bar_data)}本のバーデータを取得")
        
        # バックテスト実行
        tick_data = convert_bars_to_ticks(bar_data)
        
        # 戦略とエンジン設定
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30)
        engine = BacktestEngine(
            initial_capital=10_000_000,
            commission_rate=0.0003,
            slippage_rate=0.0001
        )
        
        engine.set_strategy(strategy)
        engine.run(tick_data)
        
        # 移動平均を計算
        bar_data['MA_fast'] = bar_data['close'].rolling(window=10).mean()
        bar_data['MA_slow'] = bar_data['close'].rolling(window=30).mean()
        
        # データを準備
        bar_data['time_str'] = bar_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 取引データを準備
        buy_trades = []
        sell_trades = []
        
        for trade in engine.trades:
            trade_data = {
                'time': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'price': float(trade.price),
                'quantity': trade.quantity
            }
            if trade.side == OrderSide.BUY:
                buy_trades.append(trade_data)
            else:
                sell_trades.append(trade_data)
        
        # エクイティカーブ
        equity_df = pd.DataFrame(engine.equity_curve)
        equity_df['timestamp_str'] = equity_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # パフォーマンスメトリクス
        metrics = engine.get_performance_metrics()
        
        # HTML生成
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>トヨタ自動車 バックテスト結果</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 14px;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .chart-container {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .trade-summary {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>トヨタ自動車 (7203) - バックテスト結果</h1>
        <p>期間: {START_DATE.strftime('%Y年%m月%d日')} ～ {END_DATE.strftime('%Y年%m月%d日')}</p>
        <p>戦略: 移動平均クロス (MA10 × MA30)</p>
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">総リターン</div>
            <div class="metric-value {('positive' if metrics.get('total_return', 0) >= 0 else 'negative')}">
                {metrics.get('total_return', 0):.2%}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">シャープレシオ</div>
            <div class="metric-value">
                {metrics.get('sharpe_ratio', 0):.2f}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">最大ドローダウン</div>
            <div class="metric-value negative">
                {metrics.get('max_drawdown', 0):.2%}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">総取引数</div>
            <div class="metric-value">
                {metrics.get('total_trades', 0)}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">勝率</div>
            <div class="metric-value">
                {metrics.get('win_rate', 0):.1%}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">最終資産</div>
            <div class="metric-value">
                ¥{metrics.get('final_equity', 0):,.0f}
            </div>
        </div>
    </div>
    
    <div class="chart-container">
        <h2>価格チャートと取引タイミング</h2>
        <div id="priceChart" style="width:100%; height:500px;"></div>
    </div>
    
    <div class="chart-container">
        <h2>資産推移</h2>
        <div id="equityChart" style="width:100%; height:300px;"></div>
    </div>
    
    <div class="trade-summary">
        <h2>取引サマリー</h2>
        <p>買い取引: {len(buy_trades)}回 / 売り取引: {len(sell_trades)}回</p>
        <table>
            <tr>
                <th>取引番号</th>
                <th>時刻</th>
                <th>売買</th>
                <th>価格</th>
                <th>数量</th>
            </tr>
"""
        
        # 最初の20件の取引を表示
        for i, trade in enumerate(engine.trades[:20]):
            html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{trade.timestamp.strftime('%m/%d %H:%M')}</td>
                <td class="{('positive' if trade.side == OrderSide.BUY else 'negative')}">
                    {('買い' if trade.side == OrderSide.BUY else '売り')}
                </td>
                <td>¥{trade.price:,.1f}</td>
                <td>{trade.quantity:,}株</td>
            </tr>
"""
        
        html_content += f"""
        </table>
        <p style="margin-top: 10px; color: #666;">※最初の20件を表示</p>
    </div>
    
    <script>
        // データの準備
        var barData = {json.dumps(bar_data[['time_str', 'open', 'high', 'low', 'close', 'volume', 'MA_fast', 'MA_slow']].to_dict('records'))};
        var buyTrades = {json.dumps(buy_trades)};
        var sellTrades = {json.dumps(sell_trades)};
        var equityData = {json.dumps(equity_df[['timestamp_str', 'equity']].to_dict('records'))};
        
        // 価格チャート
        var candlestick = {{
            x: barData.map(d => d.time_str),
            open: barData.map(d => d.open),
            high: barData.map(d => d.high),
            low: barData.map(d => d.low),
            close: barData.map(d => d.close),
            type: 'candlestick',
            name: '価格',
            increasing: {{fillcolor: '#3D9970', line: {{color: '#3D9970'}}}},
            decreasing: {{fillcolor: '#FF4136', line: {{color: '#FF4136'}}}}
        }};
        
        var maFast = {{
            x: barData.map(d => d.time_str),
            y: barData.map(d => d.MA_fast),
            type: 'scatter',
            mode: 'lines',
            name: 'MA(10)',
            line: {{color: 'orange', width: 2}}
        }};
        
        var maSlow = {{
            x: barData.map(d => d.time_str),
            y: barData.map(d => d.MA_slow),
            type: 'scatter',
            mode: 'lines',
            name: 'MA(30)',
            line: {{color: 'blue', width: 2}}
        }};
        
        var buyMarkers = {{
            x: buyTrades.map(d => d.time),
            y: buyTrades.map(d => d.price),
            type: 'scatter',
            mode: 'markers',
            name: '買い',
            marker: {{
                symbol: 'triangle-up',
                size: 12,
                color: 'green'
            }}
        }};
        
        var sellMarkers = {{
            x: sellTrades.map(d => d.time),
            y: sellTrades.map(d => d.price),
            type: 'scatter',
            mode: 'markers',
            name: '売り',
            marker: {{
                symbol: 'triangle-down',
                size: 12,
                color: 'red'
            }}
        }};
        
        var priceLayout = {{
            title: '',
            xaxis: {{
                rangeslider: {{visible: false}},
                type: 'date'
            }},
            yaxis: {{
                title: '価格 (円)'
            }},
            showlegend: true,
            height: 500
        }};
        
        Plotly.newPlot('priceChart', [candlestick, maFast, maSlow, buyMarkers, sellMarkers], priceLayout);
        
        // 資産推移チャート
        var equityTrace = {{
            x: equityData.map(d => d.timestamp_str),
            y: equityData.map(d => d.equity),
            type: 'scatter',
            mode: 'lines',
            name: '資産総額',
            fill: 'tozeroy',
            fillcolor: 'rgba(0,100,200,0.2)',
            line: {{color: 'rgb(0,100,200)', width: 2}}
        }};
        
        var initialCapital = {{
            x: equityData.map(d => d.timestamp_str),
            y: equityData.map(d => 10000000),
            type: 'scatter',
            mode: 'lines',
            name: '初期資金',
            line: {{color: 'gray', dash: 'dash'}}
        }};
        
        var equityLayout = {{
            title: '',
            xaxis: {{
                type: 'date'
            }},
            yaxis: {{
                title: '資産額 (円)',
                tickformat: ',.0f'
            }},
            showlegend: true,
            height: 300
        }};
        
        Plotly.newPlot('equityChart', [equityTrace, initialCapital], equityLayout);
    </script>
</body>
</html>
"""
        
        # ファイル保存
        filename = f'reports/toyota_backtest_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        os.makedirs('reports', exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[INFO] HTMLチャートを保存: {filename}")
        print(f"[INFO] ブラウザで開いてください: file:///{os.path.abspath(filename)}")
        
        # 自動的にブラウザで開く
        import webbrowser
        webbrowser.open(f'file:///{os.path.abspath(filename)}')
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        fetcher.disconnect()


if __name__ == "__main__":
    create_interactive_chart()