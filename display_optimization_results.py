"""
最適化結果を見やすく表示
"""
import pandas as pd
import os
from datetime import datetime

# 最新のCSVファイルを探す
reports_dir = "reports"
csv_files = [f for f in os.listdir(reports_dir) if f.startswith("toyota_optimization_results_") and f.endswith(".csv")]

if not csv_files:
    print("結果ファイルが見つかりません")
    exit()

# 最新のファイルを読み込む
latest_file = sorted(csv_files)[-1]
results_df = pd.read_csv(os.path.join(reports_dir, latest_file))

print("=== トヨタ株 移動平均クロス戦略 最適化結果 ===\n")

# 全体の統計
total = len(results_df)
positive = len(results_df[results_df['total_return'] > 0])
print(f"テストした組み合わせ: {total}通り")
print(f"プラスリターン: {positive}/{total} ({positive/total*100:.1f}%)")
print(f"平均リターン: {results_df['total_return'].mean():.2%}")
print(f"平均シャープレシオ: {results_df['sharpe_ratio'].mean():.2f}")
print()

# 上位10件（シャープレシオ順）
print("=== 上位10件の組み合わせ（シャープレシオ順） ===")
print("順位 | 時間足 | MA期間 | リターン | シャープ | 最大DD | 取引数")
print("-" * 65)

top10 = results_df.nlargest(10, 'sharpe_ratio')
for i, (_, row) in enumerate(top10.iterrows()):
    print(f"{i+1:>4} | {row['timeframe']:>5}分 | {row['fast_ma']:>3}/{row['slow_ma']:<3} | "
          f"{row['total_return']:>7.2%} | {row['sharpe_ratio']:>6.2f} | "
          f"{row['max_drawdown']:>6.2%} | {row['total_trades']:>5.0f}")

# 時間足別の最適パラメータ
print("\n=== 時間足別の最適パラメータ ===")
timeframes = sorted(results_df['timeframe'].unique())

for tf in timeframes:
    tf_results = results_df[results_df['timeframe'] == tf]
    best = tf_results.nlargest(1, 'sharpe_ratio').iloc[0]
    avg_return = tf_results['total_return'].mean()
    avg_trades = tf_results['total_trades'].mean()
    
    print(f"\n{tf}分足:")
    print(f"  最適: MA{best['fast_ma']:.0f}/{best['slow_ma']:.0f} "
          f"(リターン: {best['total_return']:.2%}, シャープ: {best['sharpe_ratio']:.2f})")
    print(f"  平均: リターン {avg_return:.2%}, 取引数 {avg_trades:.1f}")

# リターン上位（実利益重視）
print("\n=== リターン上位5件 ===")
print("順位 | 時間足 | MA期間 | リターン | シャープ | 最終資産")
print("-" * 65)

top5_return = results_df.nlargest(5, 'total_return')
for i, (_, row) in enumerate(top5_return.iterrows()):
    final_equity = row['final_equity']
    print(f"{i+1:>4} | {row['timeframe']:>5}分 | {row['fast_ma']:>3}/{row['slow_ma']:<3} | "
          f"{row['total_return']:>7.2%} | {row['sharpe_ratio']:>6.2f} | "
          f"{final_equity:>12,.0f}円")

# 推奨設定
print("\n=== 推奨設定 ===")

# バランス重視（シャープレシオ > 1 かつ リターン > 1%）
balanced = results_df[(results_df['sharpe_ratio'] > 1) & (results_df['total_return'] > 0.01)]
if not balanced.empty:
    best_balanced = balanced.nlargest(1, 'sharpe_ratio').iloc[0]
    print(f"\nバランス重視:")
    print(f"  時間足: {best_balanced['timeframe']:.0f}分")
    print(f"  移動平均: {best_balanced['fast_ma']:.0f}/{best_balanced['slow_ma']:.0f}")
    print(f"  期待リターン: {best_balanced['total_return']:.2%}")
    print(f"  シャープレシオ: {best_balanced['sharpe_ratio']:.2f}")
    print(f"  取引頻度: {best_balanced['total_trades']:.0f}回/週")

# 安定重視（最大DD < 2%）
stable = results_df[results_df['max_drawdown'] > -0.02]
if not stable.empty:
    best_stable = stable.nlargest(1, 'total_return').iloc[0]
    print(f"\n安定重視（低ドローダウン）:")
    print(f"  時間足: {best_stable['timeframe']:.0f}分")
    print(f"  移動平均: {best_stable['fast_ma']:.0f}/{best_stable['slow_ma']:.0f}")
    print(f"  期待リターン: {best_stable['total_return']:.2%}")
    print(f"  最大DD: {best_stable['max_drawdown']:.2%}")

# 時間足別の特徴
print("\n=== 時間足別の特徴 ===")
for tf in timeframes:
    tf_results = results_df[results_df['timeframe'] == tf]
    positive_rate = len(tf_results[tf_results['total_return'] > 0]) / len(tf_results) * 100
    
    print(f"\n{tf}分足:")
    print(f"  プラス率: {positive_rate:.1f}%")
    print(f"  平均取引数: {tf_results['total_trades'].mean():.1f}回")
    print(f"  最高リターン: {tf_results['total_return'].max():.2%}")
    print(f"  最高シャープ: {tf_results['sharpe_ratio'].max():.2f}")

print("\n注意: これらの結果は過去1週間のデータに基づいています。")
print("実際の取引では、より長期間のバックテストを推奨します。")