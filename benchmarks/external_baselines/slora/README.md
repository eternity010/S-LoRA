# Replicated Original S-LoRA Baseline

This baseline starts three independent, unmodified S-LoRA servers and routes
the common real trace with strict client-side round robin. It is reported as
`S-LoRA x3 + client-side RR`, not as native multi-GPU S-LoRA.

## Configuration

- Repository: `/home/hzheng/S-LoRA-baseline`
- Conda environment: `/home/hzheng/.conda/envs/slora-baseline`
- GPUs: `1,2,3`
- HTTP ports: `38200,38201,38202`
- Adapters: 100 logical adapters with the same alternating rank-16/rank-64 map
- `max_total_token_num=15000`
- `batch_max_tokens=3072`
- `max_req_total_len=2048`

Each invocation starts fresh servers and terminates all three process groups.

## Run 6 RPS

```bash
cd /home/hzheng/S-LoRA/benchmarks

/home/hzheng/.conda/envs/slora/bin/python \
  external_baselines/slora/run_replicated_realtrace.py \
  --trace-file real_workload/outputs/azure_llm_http_top100_6rps_180s_modelctx2048_v1.jsonl \
  --output-dir routing_comparison_results/post_state_fix_v1/08_external_baselines/slora_original_3replica_6rps \
  --run-label run1 \
  --duration 180
```
