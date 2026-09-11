# A100 40GB GRPO 运行记录

## 目标配置

- 8 × NVIDIA A100-SXM4-40GB
- Qwen3.5-2B SFT stage-C merged 模型
- 最大 prompt 4,096 tokens，最大 response 20,480 tokens
- 最大总上下文 24,576 tokens
- prompt batch 8，每个 prompt 4 条 rollout
- 每卡 micro batch 1

## 2026-09-12 OOM

首次 8 卡运行使用 FSDP、CPU 参数/优化器 offload、24,576-token 上下文和
`rollout.gpu_memory_utilization=0.30`。Rollout 阶段每卡约占 12.8 GiB；第一次
有效 actor 更新在 `loss.backward()` 中失败：

```text
torch.OutOfMemoryError: Tried to allocate 5.44 GiB.
GPU 0 total capacity: 39.49 GiB
GPU 0 free: 5.02 GiB
process memory in use: 33.75 GiB
PyTorch allocated: 32.55 GiB
```

该错误发生在 actor 反向传播阶段，而不是 vLLM rollout 阶段。由于每卡 micro
batch 已为 1，仅降低 mini batch 不足以可靠降低单条长序列的激活峰值；只降低
vLLM 显存比例同样不能提供足够安全余量。

## 40GB 修订策略

保留 24,576-token 上下文，并在 `a100-40gb-8x` profile 中启用：

```text
actor_rollout_ref.model.use_remove_padding=true
actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=2
actor_rollout_ref.rollout.gpu_memory_utilization=0.20
actor_rollout_ref.rollout.max_num_seqs=2
```

Ulysses 将单条长序列跨两张 GPU 处理；8 张卡形成 4 个两卡序列并行组。
`use_remove_padding=true` 是 veRL FSDP Ulysses 的运行要求。vLLM 的预算和并发
同步降低，为 actor 反向传播保留安全余量。

ShopSimulator 应至少提供 32 个槽位，与 `8 prompts × 4 rollouts` 的名义并发
一致：

```bash
SHOPSIM_PORT=5700 SHOPSIM_ENV_SLOTS=32 bash scripts/start_environment.sh
```

正式长跑前应确认至少一个 actor optimizer update 成功，并记录单卡峰值显存、
单步耗时、动态采样重试次数和有效轨迹比例。
