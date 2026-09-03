# Step-4 最小推理

> English version: [README.md](README.md)

本目录包含独立的 Step-4 推理路径,已在单节点 **8 × NVIDIA H200**、张量并行 **TP=8**
上验证。

默认的滑窗注意力走 PyTorch SDPA。QKNorm/RoPE、DSA 稀疏注意力、分块缩放 FP8 专家
矩阵乘法,以及关键的 MoE 基础算子用 Triton/CUDA 实现;模型编排与 NCCL 集合通信仍是
PyTorch。

同样的 8 个进程编号（rank）承担两种角色:稠密主干做张量并行(`TP=8`),而 352 个路由专家
按专家分片铺在这些进程上(`EP=8`,每个进程 44 个专家)。这是 TP/EP **共置**拓扑,而非独立的
`TP × EP` 进程网格。

常用术语:

| 缩写 | 含义 |
|------|------|
| TP | Tensor Parallelism，张量并行，把同一层计算拆到多张卡上。 |
| EP | Expert Parallelism，专家并行，把 MoE 专家分布到多张卡上。 |
| MoE | Mixture of Experts，混合专家网络，每个 token 只选择部分专家参与计算。 |
| FP8 | 8 位浮点数格式，用于降低模型权重和计算开销。 |
| BF16 | Brain Floating Point 16，16 位浮点数格式。 |
| SDPA | PyTorch 的缩放点积注意力实现。 |
| DSA | 稀疏注意力实现，先选出最相关的历史区域，再计算注意力。 |
| QKNorm | 对注意力中的 query 和 key 向量做归一化。 |
| RoPE | Rotary Position Embedding，旋转位置编码，用于表示 token 的相对位置。 |
| NCCL | NVIDIA 集合通信库，用于多张 GPU 之间的数据交换。 |
| KV cache | 保存历史 token 的注意力 key/value，避免解码时重复计算。 |
| MTP | Multi-Token Prediction，多 token 预测层，可用于辅助投机解码。 |
| token | 模型处理文本时使用的基本单位。 |

## 目录内容

| 文件 | 是什么 |
|------|--------|
| `config.json` | 经验证的 TP=8 参考模型配置(用 Step-4 自己的字段名,带 `gqa-provider-shared-tp-v2` 布局标记)。 |
| `convert.py` | 把原始 Step-4 模型权重转换成按进程分片的 TP 权重:`tp8/model-r{0..7}.safetensors` + 对应的 `config-r{0..7}.json`,外加一份共享的 `tokenizer_files/`。 |
| `generate.py` | 贪心推理入口。每个 `torchrun` 进程加载一个分片,调用 `Step4ForCausalLM.generate_greedy`;可传文本 `--prompt`,或通过 `--prompt-json` / `--prompt-json-batch` 传精确的 token 编号。 |
| `model.py` | Step-4 模型定义 —— 配置、解码层、注意力(full-attention 层走 DSA,滑窗层走 SDPA)以及 TP/EP 层的接线。计算算子从 `kernel.py` 导入。生成文件,请勿手改。 |
| `kernel.py` | Triton/CUDA 计算算子:稀疏注意力(DSA)索引器 + 解码/预填充、分块缩放 FP8 专家矩阵乘法、MoE 基础算子,以及融合的 QKNorm/RoPE。生成文件,请勿手改。 |
| `requirements.txt` | 锁定的运行时依赖(torch 2.10.0、transformers 4.57.6、safetensors 0.7.0、triton 3.6.0)。 |

典型流程:`convert.py`(一次性,把模型权重分片)→ `generate.py`(运行)。
`model.py` + `kernel.py` 是模型本身,由 `generate.py` 导入。

## 安装

使用与主机驱动匹配的、启用 CUDA 的 PyTorch 构建。锁定的基线环境是
PyTorch 2.10.0+cu128、Triton 3.6.0、Transformers 4.57.6、Safetensors 0.7.0；
发布版也已在[评测环境清单](../evaluation/ENVIRONMENT.md)所记录的 cu129 干净镜像中回归。

发布版直接读取本地 `tokenizer.json` 中保存的快速分词规则。Transformers 5.15
若只按旧的分词器类别信息自动选择实现,会错误地把长文本压缩成很少的 token。
字节级 BPE 会先把文本表示成字节,再把常见的连续字节合并成文本片段;直接读取
`tokenizer.json` 可以同时保留 Transformers 4.57.6 下验证过的输入 token 编号和
正常的输出文本还原。

```bash
pip install -r requirements.txt
```

## 转换模型权重

在本目录下运行。转换会在 `SAVE_PATH/tp8` 下为每个进程写一份权重和配置文件,并把
分词器文件复制到 `SAVE_PATH/tokenizer_files`。

```bash
export HF_CKPT_PATH=/path/to/Step-4
export SAVE_PATH=/path/to/Step-4-TP8

python3 convert.py \
  --checkpoint "$HF_CKPT_PATH" \
  --out-dir "$SAVE_PATH" \
  --tp-size 8 \
  --ep-size 8
```

`config.json` 是经验证的 TP=8 参考配置。`convert.py` 读取源模型权重的配置,写出匹配的
`config-r{0..7}.json`,带上必需的 `gqa-provider-shared-tp-v2` 布局标记和
`expert_parallel_size=8`;不要把它们和另一份模型权重或旧 TP 布局的权重混用。逐字节固定的
参考 `config.json` 早于那份显式 EP 元数据,所以 `generate.py` 会把缺失的
`expert_parallel_size` 当作等于 `tp_size`。

## 生成

转换时复制的分词器会被自动发现。

```bash
torchrun --standalone --nproc-per-node=8 generate.py \
  --tp-dir "$SAVE_PATH/tp8" \
  --ep-size 8 \
  --prompt "请介绍一下张量并行。" \
  --max-new-tokens 128
```

若要传精确的 token 编号,给一个 JSON 列表或 `{"prompt_ids": [...]}`:

```bash
torchrun --standalone --nproc-per-node=8 generate.py \
  --tp-dir "$SAVE_PATH/tp8" \
  --ep-size 8 \
  --prompt-json /path/to/prompt.json \
  --max-new-tokens 128
```

## 限制

- 只验证过单节点、共置 `TP=EP=8` 的 8 × H200。
- `--ep-size` 默认 8,且当前必须等于 TP / 进程总数;独立的 TP 与 EP 进程组(例如
  `TP=4, EP=2`)未实现。`generate.py` 从分片配置读取 TP(并对进程总数 `WORLD_SIZE` 做校验),
  刻意没有单独的 `--tp-size` 开关。
- 滑窗注意力用验证过的 SDPA 路径;实验性的纯 Triton 滑窗后端刻意未包含在这个最小发布版里。
- 只支持贪心生成。连续动态批处理、分页 KV 缓存、前缀缓存、CUDA Graph 和多节点执行
  均未提供。
- MTP 层不加载,因此没有投机解码。
- 直接使用原始模型权重中的 FP8/BF16 专家布局;转换器与生成的 TP 分片和该权重布局绑定。
