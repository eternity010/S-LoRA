"""
从各 suite 子目录的 results.jsonl 中提取 w2 搜索数据，
生成汇总 CSV 方便画图。

目录结构：
  routing_comparison_results/
    rwpt-w2-search/results.jsonl
    tc-w2-search/results.jsonl
    ql-w2-search/results.jsonl   (如果已跑)
    w2_search_data/
      w2_search_summary.csv
"""

import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "w2_search_data")

# suite 名称 -> load_metric 标签
SUITE_MAP = {
    "rwpt-w2-search": "rwpt",
    "tc-w2-search": "token_count",
    "ql-w2-search": "queue_length",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 从各 suite 子目录读取数据
groups = {}
for suite_name, metric_label in SUITE_MAP.items():
    result_file = os.path.join(BASE_DIR, suite_name, "results.jsonl")
    if not os.path.exists(result_file):
        print(f"  {suite_name}: not found, skipping")
        continue

    data = []
    with open(result_file) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    groups[metric_label] = data
    print(f"  {suite_name} ({metric_label}): {len(data)} experiments")

if not groups:
    print("No data found in any suite subdirectory.")
    exit(1)

print(f"\nTotal metrics loaded: {len(groups)}")

# 生成汇总 CSV
csv_file = os.path.join(OUTPUT_DIR, "w2_search_summary.csv")
fields = [
    "load_metric", "w2", "throughput", "strip_throughput",
    "avg_latency", "avg_first_token_latency",
    "p50_latency", "p90_latency",
    "p50_first_token_latency", "p90_first_token_latency",
    "cache_hit_rate", "total_requests", "cache_hits", "cache_misses",
]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for metric in ["rwpt", "token_count", "queue_length"]:
        if metric not in groups:
            continue
        for r in sorted(groups[metric], key=lambda x: x["config"]["routing_w2"]):
            row = {
                "load_metric": metric,
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

# 打印各 metric 最优点
print("\n=== Best w2 per metric (by P90 latency, lower is better) ===")
for metric in ["rwpt", "token_count", "queue_length"]:
    if metric not in groups:
        continue
    best = min(groups[metric], key=lambda r: r["result"]["p90_latency"])
    c = best["config"]
    res = best["result"]
    print(f"  {metric:14s} | w2={c['routing_w2']:4.2f} | "
          f"tput={res['throughput']:.3f} | p90={res['p90_latency']:.2f}s | "
          f"cache={res['cache_hit_rate']:.1%}")

print("\n=== Best w2 per metric (by throughput, higher is better) ===")
for metric in ["rwpt", "token_count", "queue_length"]:
    if metric not in groups:
        continue
    best = max(groups[metric], key=lambda r: r["result"]["throughput"])
    c = best["config"]
    res = best["result"]
    print(f"  {metric:14s} | w2={c['routing_w2']:4.2f} | "
          f"tput={res['throughput']:.3f} | p90={res['p90_latency']:.2f}s | "
          f"cache={res['cache_hit_rate']:.1%}")
