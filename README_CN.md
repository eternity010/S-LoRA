# S-LoRA: 服务数千个并发 LoRA 适配器 [[论文](https://arxiv.org/abs/2311.03285)]

<p align="center">
<img src="figures/serving_perf.png" alt="perf" width="700"/>
</p>

---

*最新消息*
- 公平调度器 VTC ([论文](https://arxiv.org/abs/2401.00588)) 已集成到 S-LoRA 中。
  请查看文件 `slora/server/router/vtc_req_queue.py`。

---

## 摘要
"预训练-微调"范式在大语言模型的部署中被广泛采用。低秩适应（LoRA）作为一种参数高效的微调方法，通常用于将基础模型适配到多个任务，从而产生大量从同一个基础模型衍生出的 LoRA 适配器。我们观察到这种范式在服务过程中为批处理推理提供了重要机会。为了利用这些机会，我们提出了 S-LoRA，一个专为可扩展服务多个 LoRA 适配器而设计的系统。S-LoRA 将所有适配器存储在主内存中，并将当前运行查询使用的适配器提取到 GPU 内存中。为了高效使用 GPU 内存并减少碎片化，S-LoRA 提出了统一分页（Unified Paging）。统一分页使用统一的内存池来管理具有不同秩的动态适配器权重和具有不同序列长度的 KV 缓存张量。此外，S-LoRA 采用了一种新颖的张量并行策略和高度优化的自定义 CUDA 内核，用于 LoRA 计算的异构批处理。这些特性共同使 S-LoRA 能够在单个 GPU 或多个 GPU 上以较小的开销服务数千个 LoRA 适配器。与最先进的库（如 HuggingFace PEFT 和 vLLM（对 LoRA 服务的原生支持））相比，S-LoRA 可以将吞吐量提高多达 4 倍，并将服务的适配器数量增加几个数量级。因此，S-LoRA 能够可扩展地服务许多任务特定的微调模型，并为大规模定制微调服务提供了潜力。

<p align="center">
<img src="figures/overview.png" alt="overview" width="500"/>
</p>

## 系统要求
* 兼容 CUDA 11.8 的 GPU
  * 推荐：来自 Ampere 系列的 GPU，如 A100，支持 bfloat16 操作。
  * 注意：不支持来自 Turing 系列的老旧 GPU（如 T4），因为它们不支持 bfloat16。
* 1.13 <= PyTorch <= 2.0.1

## 安装
```bash
conda create -n slora python=3.9
conda activate slora 
# 可选：通过 conda 安装 CUDA 以获得更流畅的安装体验，
# 但您可能需要手动设置 Anaconda 路径变量。
# conda install cuda -c nvidia/label/cuda-11.8.0
# 设置环境变量：export TORCH_CUDA_ARCH_LIST="8.0 8.6"
pip install torch==2.0.1
pip install -e .
```
确保 triton==2.1.0

有关通过 conda 安装 CUDA 的更多详细信息，请参阅 [NVIDIA 的 CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation)。

## GPU 设置
使用 `CUDA_VISIBLE_DEVICES` 环境变量指定使用的 GPU：

```bash
# 使用 GPU 0 和 1
export CUDA_VISIBLE_DEVICES=2
cd benchmarks
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real
```

## 运行示例
真实模型权重
```bash
cd benchmarks
python launch_server.py --num-adapter 100 --num-token 10000 --model-setting Real
python run_exp.py --debug --model-setting Real
```

虚拟权重
```bash
cd benchmarks
python launch_server.py --num-adapter 100 --num-token 10000 --dummy
python run_exp.py --debug
```

测试
```bash
cd test/test_e2e
python launch_server.py
python run_exp.py
```

## 方法

- **统一分页（Unified Paging）**：为了减少内存碎片并增加批处理大小，S-LoRA 引入了统一内存池。该池通过统一分页机制管理动态适配器权重和 KV 缓存张量。

<p align="center">
<img src="figures/unifiedpaging.png" alt="unifiedpaging" width="400"/>
</p>

- **异构批处理（Heterogeneous Batching）**：为了最小化在批处理不同秩的适配器时的延迟开销，S-LoRA 采用了高度优化的自定义 CUDA 内核。这些内核直接在非连续内存上操作，并与内存池设计对齐，为添加的 LoRA 计算提供高效的批处理推理。

- **S-LoRA TP**：为了确保跨多个 GPU 的有效并行化，S-LoRA 引入了一种新颖的张量并行策略。与基础模型相比，这种方法为添加的 LoRA 计算产生的通信成本最小。这是通过在小的中间张量上调度通信并将它们与基础模型的通信融合来实现的。

<p align="center">
<img src="figures/slora_tp.png" alt="slora_tp" width="900"/>
</p>

## 评估

### 设置

模型设置：
| 设置 | 基础模型 | 隐藏层大小 | 适配器秩 |
|---|---|---|---|
| S1 | Llama-7B | 4096 | {8} |
| S2 | Llama-7B | 4096 | {64, 32, 16, 8} |
| S4 | Llama-13B | 5120 | {64, 32, 16} |
| S5 | Llama-30B | 7168 | {32} |
| S6 | Llama-70B | 8192 | {64} |

基线：

PEFT 代表 HuggingFace PEFT：我们使用它构建了一个服务器，该服务器批处理单个适配器请求并在批次之间切换适配器权重。

vLLM-packed：由于 vLLM 不支持 LoRA，我们将 LoRA 权重合并到基础模型中，并分别服务合并权重的多个版本。为了服务 m 个 LoRA 适配器，我们在单个 GPU 上运行 m 个 vLLM 工作进程，其中多个工作进程是由 NVIDIA MPS 管理的独立进程。

S-LoRA-no-unify-mem：没有统一分页的 S-LoRA。

S-LoRA-bmm：没有统一分页和自定义内核的 S-LoRA。它将适配器权重复制到连续内存空间，并使用填充执行批处理矩阵乘法。

有关合成工作负载的跟踪信息，请参阅我们的论文。

### 结果

- 我们将 S-LoRA 与 vLLM-packed 和 HuggingFace PEFT 在服务多个 LoRA 适配器方面进行了比较。

<p align="center">
<img src="figures/vllm_and_peft.png" alt="vllm_and_peft" width="400"/>
</p>

- 与我们自己的变体进行比较。

<p align="center">
<img src="figures/synthetic.png" alt="synthetic" width="800"/>
</p>

- 我们测试了张量并行策略的可扩展性。

<p align="center">
<img src="figures/tp.png" alt="tp" width="600"/>
</p>

## 致谢
S-LoRA 构建在 [LightLLM](https://github.com/ModelTC/lightllm.git) 之上。

在开发 S-LoRA 时，我们也从以下项目中学习了很多。
- [punica](https://github.com/punica-ai/punica.git)
- [PEFT](https://github.com/huggingface/peft.git)
- [vLLM](https://github.com/vllm-project/vllm)

## 路线图
- [ ] 发布张量并行实现
- [ ] 清理可复现脚本
- [ ] 更用户友好的 API/前端
- [ ] 更多模型支持

## 引用
```bibtex
@article{sheng2023slora,
  title={S-LoRA: Serving Thousands of Concurrent LoRA Adapters},
  author={Sheng, Ying and Cao, Shiyi and Li, Dacheng and Hooper, Coleman and Lee, Nicholas and Yang, Shuo and Chou, Christopher and Zhu, Banghua and Zheng, Lianmin and Keutzer, Kurt and Gonzalez, Joseph E. and Stoica, Ion},
  journal={arXiv preprint arXiv:2311.03285},
  year={2023}
}
```
```bibtex
@article{sheng2023fairness,
  title={Fairness in Serving Large Language Models},
  author={Sheng, Ying and Cao, Shiyi and Li, Dacheng and Zhu, Banghua and Li, Zhuohan and Zhuo, Danyang and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2401.00588},
  year={2023}
}
```

