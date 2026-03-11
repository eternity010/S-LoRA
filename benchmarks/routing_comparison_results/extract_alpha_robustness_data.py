"""
从 results.jsonl 中提取最新一轮 alpha-robustness-sla 实验数据（最后 9 条），
生成按 alpha 分组的 jsonl 文件和汇总 CSV。

实验设计：3 alpha × 3 metric = 9 experiments
  w2 选取：RWPT 和 TC 量纲一致，统一 w2=1.0；QL 无归一化，w2=0.15
  rwpt:         w2=1.0
  token_count:  w2=1.0
  queue_length: w2=0.15

注意：results.jsonl 中有多轮 alpha 实验（v1 旧 w2, v2 TC=3.5, v3 TC=1.0），
本脚本只取最后 9 条（v3，最终版本）。
"""

import json
import csv
import os

RESULTS_FILE = "results.jsonl"
OUTPUT_DIR = "alpha_robustness_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取全部数据
rows = []
with open(RESULTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Total rows in results.jsonl: {len(rows)}")

# 取最后 9 条（alpha-robustness-sla v3）
alpha_rows = rows[-9:]

# 验证数据正确性
expected_pairs = {
    ("rwpt", 1.0),
    ("token_count", 1.0),
    ("queue_length", 0.15),
}
expected_alphas = {0.1, 0.3, 0.8}

for r in alpha_rows:
    cfg = r["config"]
    pair = (cfg["load_metric"], cfg["routing_w2"])
    assert pair in expected_pairs, f"Unexpected (metric, w2): {pair}"
    assert cfg["alpha"] in expected_alphas, f"Unexpected alpha: {cfg['alpha']}"

print(f"Extracted {len(alpha_rows)} experiments (v3: RWPT=1.0, TC=1.0, QL=0.15)")

# 按 alpha 分组
by_alpha = {}
for r in alpha_rows:
    alpha = r["config"]["alpha"]
    by_alpha.setdefault(alpha, []).append(r)

# 输出 results.jsonl（覆盖旧版）
out_results = os.path.join(OUTPUT_DIR, "results.jsonl")
with open(out_results, "w") as f:
    for r in alpha_rows:
        f.write(json.dumps(r) + "\n")
print(f"  -> {out_results}")

# 输出每个 alpha 的 jsonl 文件
for alpha in sorted(by_alpha):
    out_file = os.path.join(OUTPUT_DIR, f"alpha_{alpha}_results.jsonl")
    with open(out_file, "w") as f:
        for r in by_alpha[alpha]:
            f.write(json.dumps(r) + "\n")
    print(f"  alpha={alpha}: {len(by_alpha[alpha])} experiments -> {out_file}")

# 生成汇总 CSV
csv_file = os.path.join(OUTPUT_DIR, "alpha_robustness_summary.csv")
fields = [
    "alpha", "load_metric", "w2", "throughput", "strip_throughput",
    "avg_latency", "avg_first_token_latency",
    "p50_latency", "p90_latency",
    "p50_first_token_latency", "p90_first_token_latency",
    "cache_hit_rate", "total_requests", "cache_hits", "cache_misses",
]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for alpha in [0.1, 0.3, 0.8]:
        for r in by_alpha[alpha]:
            row = {
                "alpha": alpha,
                "load_metric": r["config"]["load_metric"],
                "w2": r["config"]["routing_w2"],
                "throughput": round(r["result"]["throughput"], 4),
                "strip_throughput": round(r["result"]["strip_throughput"], 4),
                "avg_latency": round(r["result"]["avg_latency"], 3),
                "avg_first_token_latency": round(r["result"]["avg_first_token_latency"], 3),
                "p50_latency": round(r["result"]["p50_latency"], 3),
                "p90_latency": round(r["result"]["p90_latency"], 3),
                "p50_first_token_latency": round(r["result"]["p50_first_token_latency"], 3),
                "p90_first_token_latency": round(r["result"]["p90_first_token_latency"], 3),
                "cache_hit_rate": round(r["result"]["cache_hit_rate"], 4),
                "total_requests": r["result"]["total_requests"],
                "cache_hits": r["result"]["cache_hits"],
                "cache_misses": r["result"]["cache_misses"],
            }
            writer.writerow(row)

print(f"\nSummary CSV -> {csv_file}")

# 打印对比表
print("\n=== Alpha Robustness Summary (RWPT=1.0, TC=1.0, QL=0.15) ===")
print(f"{'Alpha':<7} {'Metric':<15} {'w2':<6} {'Tput':<8} {'Avg Lat':<9} {'P90':<8} {'Cache':<8}")
print("-" * 65)
for alpha in [0.1, 0.3, 0.8]:
    for r in by_alpha[alpha]:
        cfg = r["config"]
        res = r["result"]
        print(f"{alpha:<7} {cfg['load_metric']:<15} {cfg['routing_w2']:<6} "
              f"{res['throughput']:<8.3f} {res['avg_latency']:<9.2f} "
              f"{res['p90_latency']:<8.2f} {res['cache_hit_rate']:<8.1%}")
    if alpha != 0.8:
        print()
