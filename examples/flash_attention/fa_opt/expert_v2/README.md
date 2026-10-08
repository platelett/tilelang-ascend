# Expert v2：三槽在线 Softmax 优化历程

本文记录
[`kernel.py`](kernel.py)
这条 Expert-mode 实现怎样形成。它是一份具体实现的 case study；通用的 CV 融合方法仍见
[性能优化指南](../docs/optimization_guide_zh.md)。

## 范围与证据边界

这个 kernel 实现非因果、FP16 输入输出的 BHSD Flash Attention，固定 `D=128`，要求
序列长度为 1024 的整数倍，支持 `heads_q % heads_kv == 0` 的 MHA/GQA。
它把 GQA 的 query group 折进逻辑 query 序列，
不复制 K/V。

实现最初在独立 FA 研究仓中演进，随后在 TileLang-Ascend PR #1852 的提交
`77a444b2` 上改用普通 FP32 `reduce_max` / `reduce_sum` 并完成最终对照。PR #1852 已合入
`ascendc_pto` 主线；本目录中的代码是该已选实现的独立单文件版本。

下文标为“历史测量”的数字来自当时同一台 Atlas A3 上的对照，不冒充当前主线的重新测量。
原始内部日志和诊断分支不随本示例发布，因此只保留整理记录能够支持的结论及其限制。

## 1. Motivation：让完整 attention 形成并行流水

长序列 attention 不能保存完整分数矩阵。目标是让 Cube、Vector 和 GM 搬运同时推进：

```text
Cube:   QK -> workspace S                  PV -> workspace O_partial
Vector:        online softmax -> P               update O_acc -> output
```

每个 key 块都要更新运行最大值、指数和、缩放系数以及输出累加。workspace 槽过早复用，
会覆盖另一个执行侧尚未消费的数据；槽太少则会让本可并行的工作互相等待。

## 2. Expectation：分开验证块大小与流水深度

最初有两个假设：更宽的 key/query 块可能减少循环与同步；如果 Cube 与 Vector 的工作量
接近，增加一个 workspace 槽可能减少交接气泡。两者分别测试，避免把块大小、同步协议和
槽数同时改变后再猜测收益来源。

## 3. Change：最终实现结构

最终版本采用：

- `TILE_Q_L2=256`，`TILE_KV_L2=1024`，`D=128`；
- 24 个逻辑 Cube core，与配对的 Vector subcore 协作；
- 三个 S/P 与 partial-O workspace 槽；
- Expert-mode 显式 C/V scope、片上地址和本地/跨核 token 生命周期；
- FP32 在线 softmax：`reduce_max -> subtract -> exp -> reduce_sum`；
- FP32 运行统计量、partial-O 工作区与 O 累加；S/P 工作区和最终输出保持 FP16；
- QK 使用 output-split MMA，PV 使用 contraction-split MMA；
- GQA 通过 `decl_buffer` 视图折叠为 MHA 形状，不复制数据。

`reduce_max` 与 `reduce_sum` 共用显式 `uint8` UB scratch。三代缩放系数与 query 统计量使用
独立 ring，避免不同流水代之间相互覆盖。

## 4. Result：关键对照

### 4.1 扩大块没有成为优化

历史实验把 key 块从 1024 扩到 2048，也增加过每块 query 行数。预期是少做循环；结果却是
更宽 key 块约慢 5%–8%，更多 query 行大体持平，更大的组合还受到片上容量与接口限制。
因此最终保留 `256 x 1024` 的通信粒度。

这组结果来自旧编译/算子环境，只支持“没有采用更大块”这个版本决策，不用于预测任意新硬件
或新编译器上的百分比。

### 4.2 消融揭示的是重叠，不是独立成本

诊断版本分别去掉 Cube、Vector、搬运、softmax 或输出更新，只保留完成协议所需的信号；
它们的输出不是正确 attention。早期聚合结果中，Cube-only 和 Vector-only 分别约为完整版本
用时的 79% 和 81%。

这不表示两部分各自只占 79% 或 81%。它说明两条流水原本有明显重叠，同时运行时还会共享
搬运资源。删除一项工作所得的时间差，不能直接当成该项工作的独立成本。

### 4.3 三个槽胜过两个，第四个没有明确收益

旧融合-softmax 阶段，三个槽相对两个槽在三个规模上约快 3%–4%，四个槽没有继续得到明确
收益。归约接口换成 PR #1852 的普通操作后，又重新比较了相同实现的两个槽和三个槽：

| Batch / Length | 两个槽，ms | 三个槽，ms | 用时下降 |
|---|---:|---:|---:|
| 2 / 131072 | 1148.701 | 1125.264 | 2.04% |
| 2 / 65536 | 285.076 | 276.667 | 2.95% |
| 1 / 32768 | 35.650 | 34.511 | 3.20% |

三组历史对照均通过当时的正确性检查，测试编译器基线为 `77a444b2`。它们支持选择三个槽，
但不是当前 `ascendc_pto` HEAD 的性能声明。

### 4.4 从融合 softmax 切换到普通归约

早期版本依赖 `SoftmaxFlashV2`。为了只使用主线接口，最终改成 FP32 `reduce_max`、减法、
`exp` 与 `reduce_sum` 的组合，同时保留跨块运行统计和延迟 O 累加。

这个改变会影响 Vector 工作量和 UB scratch，因此没有沿用旧槽数结论，而是在新接口上重做
2-vs-3 槽对照。最终选择是“普通归约 + 三槽”，不是把旧融合算子的成绩直接移植过来。

## 5. 未采用或仅用于诊断的尝试

| 尝试 | 处置 |
|---|---|
| key 块加宽到 2048 | 历史对照慢约 5%–8%，未采用 |
| 增加 query 行数 | 大体持平；更大组合受容量/接口限制，未采用 |
| 去掉 Cube、Vector、搬运或输出更新 | 仅用于理解重叠，输出无效，不作为优化版本 |
| 两槽、三槽、四槽 | 两次独立阶段都选择三槽；四槽无明确继续收益 |
| FP16 融合 softmax 与融合缩放/转换 | 当时接口不能支持所需完整组合，未成为可运行候选 |
| `SoftmaxFlashV2` 路线 | 被主线普通归约版本替代 |

## 6. 当前结论与复现

这条路线保留的是完整在线 attention、三槽所有权、显式 Expert 同步，以及已经合入主线的
普通 FP32 归约接口。它作为一条可读、可运行的优化谱系保留，不取代目录中已有的其他
Expert/Hybrid 实现，也不声称对所有 shape 都最快。

运行一个较小的正确性用例：

```bash
source ./set_env.sh
python examples/flash_attention/fa_opt/expert_v2/kernel.py \
  --B 1 --S 4096 --q-heads 12 --kv-heads 1 --D 128
```

要比较长序列性能，可让 `fa_opt/run.py` 通过 `--tl` 指向该文件。记录结果时应同时保存
TileLang 提交、shape、SoC、CANN 版本、正确性阈值和原始 timing 样本。
