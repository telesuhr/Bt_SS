# Windows環境でのセットアップ手順

## 1. 仮想環境の作成と有効化
```powershell
python -m venv venv
.\venv\Scripts\activate
```

## 2. TA-Libのインストール（Windows）

TA-Libは特殊な手順が必要です：

### 方法1: 非公式Wheelファイルを使用（推奨）
1. https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib にアクセス
2. Python 3.11用のTA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl をダウンロード
3. ダウンロードしたファイルをインストール：
```powershell
pip install TA_Lib-0.4.28-cp311-cp311-win_amd64.whl
```

### 方法2: talib-binaryパッケージを使用
```powershell
pip install talib-binary
```

## 3. その他のパッケージのインストール
```powershell
pip install -r requirements_windows.txt
```

## 4. Bloomberg API（オプション）

Bloomberg APIを使用する場合：
1. Bloomberg Terminalがインストールされていることを確認
2. Bloomberg API (BLPAPI)を有効化
3. 以下のコマンドでインストール：
```powershell
pip install blpapi
```

## 5. Redis（オプション）

Redisキャッシュを使用する場合：
1. Windows用Redis: https://github.com/microsoftarchive/redis/releases
2. またはDocker版を使用：
```powershell
docker run -d -p 6379:6379 redis:latest
```

## 6. 環境変数の設定
```powershell
copy .env.example .env
# .envファイルを編集して設定
```

## トラブルシューティング

### TA-Libのインストールエラー
- Visual Studio Build Toolsが必要な場合があります
- https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

### psycopg2のエラー
PostgreSQLを使用しない場合は、requirements_windows.txtから削除可能です。

### numbaのエラー
高速化が不要な場合は削除可能です。