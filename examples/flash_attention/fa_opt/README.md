# Flash Attention 优化实现

这个目录保留几条彼此独立的 Flash Attention 实现路线，以及统一的运行和性能对比入口。
目录按“设计谱系”组织；具体版本的说明跟随 kernel，共用的测试口径和调优方法放在 `docs/`。

## 实现

| 路线 | 入口 | 定位 |
|---|---|---|
| Expert v1 | [`expert_v1/kernel.py`](expert_v1/kernel.py) | 原有手工同步 Expert 实现；历史成绩见 benchmark 文档 |
| Expert v2 | [`expert_v2/kernel.py`](expert_v2/kernel.py) | 三槽在线 softmax Expert 实现；[优化历程](expert_v2/README.md) 与源码同目录 |
| Auto pipeline | [`auto_pipeline/`](auto_pipeline/) | H16/D128 与 H32/D512 两个自动流水版本 |
| AscendC reference | [`reference/ascendc.py`](reference/ascendc.py) | 性能对比使用的原生算子入口 |

Expert v2 是新增的实现谱系，不在没有同环境 A/B 的情况下覆盖 v1 的历史性能结论。

## 运行

从仓库根目录激活当前 TileLang 环境：

```bash
source ./set_env.sh
python examples/flash_attention/fa_opt/expert_v2/kernel.py
```

批量对比入口：

```bash
bash examples/flash_attention/fa_opt/bench.sh
```

`run.py` 默认选择 Expert v2 和 AscendC reference，也可以用 `--tl` / `--ascendc` 指定其他实现。
`examples/bench_test.sh` 会自动收集 `fa_opt` 任意一级实现目录中的 Python 入口；根目录的
`run.py` 和 `plot.py` 不会被当成 kernel 测试。

## 文档

- [历史 benchmark](docs/benchmark_zh.md)
- [通用性能优化指南](docs/optimization_guide_zh.md)
- [Expert v2 优化历程](expert_v2/README.md)
