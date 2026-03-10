"""
从 results.jsonl 中提取三个 load_metric 的 w2 搜索数据，
分离到 w2_search_data/ 目录下的独立文件。

RWPT 取第二轮数据（row 11-20），queue_length 和 token_count 取全部。
同时生成一个汇总 CSV 方便画图。
"""

import json
import csv
import os

RESULTS_FILE = "results.jsonl"
OUTPUT_DIR = "w2_search_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取全部数据
rows = []
with open(RESULTS_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Total rows: {len(rows)}")

# 按 load_metric 分组
groups = {"rwpt": [], "queue_length": [], "token_count": []}
for r in rows:
    lm = r["config"]["load_metric"]
    groups[lm].append(r)

# RWPT: 有两轮各 10 个，取第二轮（index 10-19）
rwpt_all = groups["rwpt"]
print(f"RWPT total: {len(rwpt_all)} (taking round 2: rows 11-20)")
groups["rwpt"] = rwpt_all[10:20]  # 第二轮

# 输出各 metric 的 jsonl 文件
for metric, data in groups.items():
    out_file = os.path.join(OUTPUT_DIR, f"{metric}_w2_search.jsonl")
    with open(out_file, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print(f"  {metric}: {len(data)} experiments -> {out_file}")

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
    for metric in ["rwpt", "queue_length", "token_count"]:
        for r in groups[metric]:
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
print("\n=== Best w2 per metric ===")
for metric in ["rwpt", "queue_length", "token_count"]:
    best = max(groups[metric], key=lambda r: r["result"]["throughput"])
    c = best["config"]
    res = best["result"]
    print(f"  {metric:14s} | w2={c['routing_w2']:4.2f} | "
          f"tput={res['throughput']:.3f} | lat={res['avg_latency']:.2f}s | "
          f"ftl={res['avg_first_token_latency']:.2f}s | "
          f"p90={res['p90_latency']:.2f}s | cache={res['cache_hit_rate']:.1%}")
