# Rank-dependent prefill profiling

This directory contains the scripts used to build Figure 1(a).

1. `prepare_rank_adapters.py` slices one rank-64 LoRA checkpoint into
   rank 8, 16, and 32 checkpoints. The rank-64 entry is a symlink to the
   source checkpoint, so all profiled adapters share the same layer layout.
2. `run_prefill_profile.py` starts one S-LoRA worker, warms every adapter,
   and measures time to first token for sequential one-token requests.
3. `../exported_results/plot_figure1a_rank_profile.py` fits the per-token
   prefill slope across prompt lengths and plots normalized rank overhead.

Generated checkpoints and raw results are runtime artifacts and are not
stored in Git.
