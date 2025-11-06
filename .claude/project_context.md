# Claude Code プロジェクトコンテキスト

## プロジェクト概要
Bloomberg APIを使用した日本株バックテストシステムの開発プロジェクト。

## 現在の状態（2025-11-07）

### 実装済み機能
- ✅ Bloomberg APIティックデータ取得
- ✅ バックテストエンジン（手数料・スリッページ考慮）
- ✅ 複数の取引戦略（SimpleDayTrading, MeanReversion, Breakout, Adaptive）
- ✅ 戦略最適化機能
- ✅ HTMLレポート生成
- ✅ 市場環境適応型戦略

### 主要ファイル
- `main.py` - メインエントリーポイント
- `adaptive_strategy.py` - 市場環境適応型戦略
- `find_best_strategy.py` - 最適戦略探索
- `multi_stock_daytrading.py` - 複数銘柄デイトレード

### 重要な発見
1. SimpleDayTrading戦略: トヨタで+2.70%達成
2. 最適戦略: Breakout_30_0.004（平均+0.09%）
3. 適応型戦略: ソフトバンクGで+12.20%の超過リターン

### 次回作業時の指示
```
このプロジェクトの続きをお願いします。
前回は市場環境適応型戦略まで完了しました。
次は以下のいずれかを実装したいです：
1. ポートフォリオ戦略（複数銘柄同時運用）
2. 機械学習アプローチ
3. リスク管理の高度化
```

### 技術的な注意点
- Windows環境での文字化け対策: `chcp 65001`
- Bloomberg Terminalが起動している必要あり
- 時刻表示が0-6時（実際は9-15時）になる問題あり
- `reports/`フォルダにHTMLレポートが生成される

### Bloomberg API使用例
```python
from src.bloomberg_api.tick_data_fetcher import TickDataFetcher

fetcher = TickDataFetcher()
if fetcher.connect():
    # トヨタのデータ取得
    data = fetcher.fetch_intraday_bars("7203 JT Equity", start_date, end_date, interval=1)
```

### よく使うテスト銘柄
- 7203 JT Equity (トヨタ)
- 9984 JT Equity (ソフトバンクG)
- 6758 JT Equity (ソニー)
- 8306 JT Equity (三菱UFJ)