# クイックスタートガイド

## 最速でバックテストを実行する方法

### 1. 仮想環境のセットアップ
```powershell
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化（PowerShell）
.\venv\Scripts\activate
```

### 2. 必要最小限のパッケージインストール
```powershell
# 最小構成でインストール（ta-lib不要）
pip install -r requirements_minimal.txt
```

### 3. バックテストの実行
```powershell
# デモデータでバックテスト実行
python main.py
```

## 実行結果

以下のような結果が表示されます：

```
2025-11-06 12:00:00.000 | INFO     | __main__:setup_logging:21 - === ティックデータバックテストシステム ===
2025-11-06 12:00:00.001 | INFO     | __main__:setup_logging:21 - 対象銘柄: 7203 JT Equity
2025-11-06 12:00:00.002 | INFO     | __main__:setup_logging:21 - デモデータを生成します
2025-11-06 12:00:05.000 | INFO     | __main__:setup_logging:21 - 取得データ: 77760ティック
2025-11-06 12:00:05.100 | INFO     | __main__:setup_logging:21 - 3. テクニカル指標を計算
2025-11-06 12:00:06.000 | INFO     | __main__:setup_logging:21 - 4. バックテスト実行
...
```

生成されるファイル：
- `reports/backtest_report_YYYYMMDD_HHMMSS.html` - バックテスト結果レポート
- `logs/backtest_YYYY-MM-DD_HH-MM-SS.log` - 実行ログ

## カスタマイズ

### 環境変数（.envファイル）

```bash
# Bloomberg APIを使用（デフォルト: false）
USE_BLOOMBERG=false

# データベース保存を使用（デフォルト: false）
USE_DATABASE=false

# プロット表示（デフォルト: false）
SHOW_PLOTS=true
```

### 独自戦略の追加

`src/strategies/`に新しい戦略クラスを追加：

```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def on_tick(self, tick):
        # カスタムロジック
        pass
```

## トラブルシューティング

### "ModuleNotFoundError: No module named 'src'"
→ main.pyがあるディレクトリから実行してください

### "TA-Lib not found, using lite version"
→ 正常な動作です。軽量版の指標計算ライブラリを使用します

### メモリ不足エラー
→ デモデータの期間を短くしてください（main.pyのSTART_DATE変更）