# Text2Motion–Motion Cerebellum：从文本动作生成到物理闭环跟踪

## 摘要

本项目实现并验证了一条面向 Unitree G1 的完整演示链路：自然语言首先由预训练 OMG 50M
模型生成 G1 参考动作，适配器将参考动作转换为 `motion_tracking` 的输入格式，随后由在
MuJoCo 中训练的通用动作跟踪策略闭环控制机器人。这里将低层跟踪策略称为 **Motion
Cerebellum（运动小脑）**：它接收期望动作和机器人当前状态，持续输出关节控制动作；它不
负责理解语言，也不是一个额外叠加在主模型后的平滑滤波器。

最终系统没有采用早期尝试过的 residual、preview 或 oracle 控制器，而是保留上游
`suning-git/motion_tracking` 的原始 actor 结构。我们补齐了 Text2Motion 接口、数据质量门、
正式规模训练、统一评测、三次独立从零训练以及可复现性冻结。三个训练种子在 60 条原生
held-out 动作上取得 91.67% ± 0.83% 的平均成功率，在三条文本提示生成动作上取得
94.44% ± 4.81% 的平均成功率。进一步的 12 提示预注册压力测试显示：9 条新提示中只有
3 条通过物理质量门；在 6 条通过门的提示上 tracker 平均成功率为 81.94%，但把质量门拒绝
计入后，完整端到端成功率仅为 40.97%。因此结果支持“Text2Motion 到运动小脑的可运行
demo，以及三种子正式规模基线”，但不支持开放词汇成功，也不构成真实机器人部署或对上游
论文的严格等价复现。进一步的生成阶段诊断确认：6 条质量门拒绝在 OMG 原始 30 Hz 输出中
已经存在，并非 30→50 Hz 适配造成；单段 60 帧生成的质量通过率为 6/9，而双段 120 帧为
3/9，后者 8/9 的最大关节跳变位于 59→60 分段边界。因此当前首先应修复长动作的分段衔接，
而不是立即重训运动小脑或整个 Text2Motion 模型。后续三生成种子复验进一步表明：单段质量
通过率为 21/27，双段为 9/27；全部 21 条单段通过项在三个小脑上的平均跟踪成功率为
80.56%，把 6 个生成失败计入后端到端成功率为 62.65%。这建立了短时动作 demo，但没有
修复长上下文生成。后续长动作实验把双段质量通过率经空间对齐和显式标注的接口清洗提高到
18/27，但三个小脑的平均长动作跟踪成功率仅为 70.37%，端到端为 46.91%；两次带留出生成
种子的 300-iteration 小数据微调又造成明显原生能力退化。因此长动作结论仍为负面结果。

## 1. 问题与研究问题

Text2Motion 模型擅长生成语义合理的运动轨迹，但生成轨迹并不会自动满足重力、接触、关节
执行器和状态误差下的稳定性要求。本项目研究的不是用另一个生成模型替换 Text2Motion，
而是把生成与执行分层：

```text
文本意图
  → 预训练 OMG 50M（高层动作生成）
  → G1 qpos_36 参考轨迹
  → OMG adapter（格式、频率与运动学桥接）
  → motion_tracking actor（Motion Cerebellum）
  → MuJoCo 中的闭环 G1
```

核心问题有三个：

1. OMG 输出能否无歧义地接入原有动作跟踪仓库？
2. 低层 actor 在自身 held-out 分布上足够可信后，能否稳定跟踪文本生成动作？
3. 结果是否对训练随机种子稳健，而不是某个 checkpoint 的偶然表现？

## 2. 原仓库已有内容与本项目新增内容

原始 `motion_tracking` 仓库已经提供 G1 动作跟踪 actor、参考动作格式、MuJoCo 训练与评测
流程，以及 `g1/deployable` 正式配置。它解决“给定机器人参考轨迹，如何闭环执行”的问题，
因此可直接作为运动小脑的主体。原仓库并不提供本项目所需的文本入口，也没有替我们验证
OMG 参考动作在同一评测协议下的表现。

本项目新增：

- `qpos_36 → motion_tracking reference` 的严格适配器；
- 四元数归一化与符号连续化、30→50 Hz SLERP/线性重采样、与上游一致的 `qvel_35`
  计算，以及目标 G1 MJCF 上的脚部前向运动学；
- 对参考动作姿态、速度、悬空与脚滑的质量门；
- 原生 held-out 与 OMG 文本动作的统一、带 observation noise 的 episode 级评测；
- BONES-SEED 数据诊断、resume sampling-weight 长度不匹配修复；
- 1500 条训练参考、3840 个并行环境、80 个 worker 的正式规模 clean-start 训练；
- 三个独立训练种子的统计复核和冻结清单。

## 3. 数据与接口

### 3.1 训练与测试数据

BONES-SEED 原始动作经过 SOMA→SMPL/G1 的既有处理链和项目质量门，得到 1694 条可用训练
参考。本实验对每个训练种子使用其中 1500 条；原生评测固定使用由 test-pool seed 12345
确定的 60 条 held-out 动作。训练与测试 reference shard 的最终冻结清单由
`final_freeze_v1/remote_artifact_manifest.json` 固定。

数据拆分和评测只能被解释为本项目的 held-out 协议。虽然源数据采用 actor-level 拆分思路，
本项目没有重新证明其与所有上游训练材料完全无身份或动作语义重叠，因此不将它表述为严格
跨数据集泛化。

### 3.2 Text2Motion 接口

OMG 生成 36 维 G1 状态：根节点位置 3 维、scalar-first 根四元数 4 维和 29 个关节角。
OMG 与 tracker 的 G1 关节顺序相同，所以接口不经过 SMPL 二次 retarget。适配器在关节顺序
不一致时直接拒绝转换，并将输出写成上游 `shard_*.npz` object-array reference 格式。

最终评测使用三条固定提示：向前行走、向前并左转、向前并右转。每条生成动作都先通过质量
门，再进入物理闭环。三条提示足以证明接口和基本方向控制，但不足以代表开放词汇文本生成
能力。

## 4. Motion Cerebellum 架构与训练

最终“运动小脑”就是未增加 residual 分支的上游 motion-tracking actor。高层模型给出未来
参考，actor 同时观察带噪声的机器人状态与参考状态，在 MuJoCo 中输出关节动作。训练使用：

| 项目 | 设置 |
|---|---:|
| tracker commit | `a3b8d0c092684f1307c53175d94260c4ff323306` |
| G1 model commit | `71f066ad0be9cd271f7ed58c030243ef157af9f4` |
| recipe | `g1/deployable` |
| clean start | 是，随机初始化 |
| training seeds | 0、1、2 |
| 训练参考 | 每个种子 1500 条，可用池 1694 条 |
| iterations | 4500 |
| parallel environments | 3840 |
| rollout horizon | 16 |
| workers | 80 |
| observation noise | 开启 |
| 每个种子的 transitions | 276,480,000 |

seed 1 和 seed 2 在一张 32 GB NVIDIA GeForce RTX 5090 上顺序训练与评测，共耗时
35,673 秒（9 小时 54 分 33 秒）。seed 0 来自上一轮相同冻结协议的 clean-start 训练。
最后的三种子复核、artifact 哈希、扩展提示评测和质量门诊断合计计费 59.50 元；这只代表
最终复核阶段，不代表整个项目历史成本。

## 5. 评测设计

每个训练种子都在完全相同的 episode key 上评测：

- 原生：60 条 held-out clip × 4 次独立 observation-noise repeat，共 240 episodes/seed；
- OMG：3 条文本提示 × 4 次独立 observation-noise repeat，共 12 episodes/seed；
- 三个策略合计 756 个物理仿真 episodes。

主要指标为成功率、完成率、关节相对 MPJPE（`Empjpe`）、全局 MPJPE（`Eg_mpjpe`）、
foot sliding 和 jerk。后两项沿用上游 evaluator 的数值定义，仅在完全相同的采样率、reference
和评测器下作相对比较，不把它们误写成具有通用物理单位的绝对测量。

统计复核以三个独立训练策略为 replication unit，报告均值、样本标准差和自由度 2 的双侧
t 区间。百分比区间没有裁剪到 0–100%，因此小样本 OMG 成功率的上界可以超过 100%；这反映
估计不确定性，不是概率预测。240 或 12 个 episode 不能替代三个训练种子的独立性。

预先冻结的通过门槛为：每个种子的原生成功率 ≥80%、完成率 ≥90%；每个种子的 OMG 成功率
≥2/3、完成率 ≥85%。

## 6. 主要结果

### 6.1 三种子 clean-start 结果

| Domain | Metric | seed 0 | seed 1 | seed 2 | mean ± sample SD | 95% t interval |
|---|---:|---:|---:|---:|---:|---:|
| Native | success | 91.67% | 92.50% | 90.83% | **91.67% ± 0.83%** | 89.60–93.74% |
| Native | completion | 96.24% | 96.22% | 95.56% | **96.01% ± 0.38%** | 95.05–96.96% |
| Native | Empjpe | 28.65 mm | 29.53 mm | 29.76 mm | **29.31 ± 0.58 mm** | 27.86–30.76 mm |
| Native | foot sliding | 5.270 | 5.191 | 5.276 | **5.246 ± 0.047** | 5.129–5.363 |
| Native | jerk | 5.013 | 5.151 | 5.140 | **5.101 ± 0.076** | 4.912–5.291 |
| OMG | success | 100.00% | 91.67% | 91.67% | **94.44% ± 4.81%** | 82.49–106.40% |
| OMG | completion | 100.00% | 99.62% | 99.62% | **99.75% ± 0.22%** | 99.20–100.29% |
| OMG | Empjpe | 30.39 mm | 26.87 mm | 28.30 mm | **28.52 ± 1.77 mm** | 24.13–32.91 mm |
| OMG | foot sliding | 4.669 | 4.637 | 4.939 | **4.748 ± 0.166** | 4.337–5.160 |
| OMG | jerk | 4.249 | 4.616 | 4.536 | **4.467 ± 0.193** | 3.988–4.946 |

三个训练种子全部通过预设门槛。seed 1 和 seed 2 各有一个右转 noisy repeat 失败，因而
OMG 成功率为 11/12；完成率仍为 99.62%。这说明接口和低层控制总体稳定，但右转仍是当前三条
提示中的相对薄弱项。

### 6.2 clean-start 相对旧 warm-start checkpoint

在完全配对的 60 条原生 clip × 4 repeats 上，seed-0 clean checkpoint 相对旧的 1028-clip
warm-start checkpoint，成功率从 86.7% 提高到 91.7%，foot sliding 从 5.90 降到 5.27，
jerk 从 5.45 降到 5.01。按 clip 聚类的 20,000 次 bootstrap 给出：成功率提升 2.1–8.8
个百分点、foot sliding 变化 −0.78 至 −0.49、jerk 变化 −0.70 至 −0.001 的 95% 区间。

这个比较同时改变了初始化、训练语料规模、训练种子和优化历史，因此只能说明最终 clean
baseline 更强，不能把改善单独归因于某一项改动。三种子表才是最终主结果。

### 6.3 预注册扩展提示压力测试

为检验三提示 demo 是否具有更广外推性，我们在看结果前冻结了 12 条提示和门槛：保留原
3 条提示，新增 9 条后退、侧移、慢跑、下蹲、鞠躬、手势和踢腿提示；每条只生成一次，不
允许失败后重抽。所有 9 条 OMG 生成均完成，但只有右侧移、鞠躬和右手挥手通过原质量门，
新提示通过率为 3/9（33.3%），低于预注册的 6/9 门槛。被拒绝的是后退、左侧移、慢跑、
下蹲、双臂上举和右腿前踢。

加入原有 3 条后，共 6 条 reference 进入 tracker；每个策略运行 6×4=24 个 episodes：

| Metric | seed 0 | seed 1 | seed 2 | mean ± sample SD |
|---|---:|---:|---:|---:|
| quality-passing prompt tracking success | 83.33% | 79.17% | 83.33% | **81.94% ± 2.41%** |
| quality-passing prompt completion | 90.59% | 91.56% | 95.77% | **92.64% ± 2.75%** |
| all-12-prompt end-to-end success | 41.67% | 39.58% | 41.67% | **40.97% ± 1.20%** |

鞠躬和右手挥手在三个策略、所有噪声重复上均成功，证明小脑可以跟踪原三提示以外的动作。
但右侧移的成功率依次为 0%、0%、25%，说明侧向动态仍是 tracker 的具体弱点。更重要的是，
6 条生成动作在进入小脑前就被质量门拒绝。扩展实验因此未通过完整 demo 门槛：主要瓶颈从
“小脑是否会跟踪”转向“高层生成能否稳定产生物理可接受 reference”，同时保留侧向跟踪
这一低层问题。

我们随后用固定的上游质量门重放全部 9 条新动作，得到以下拒绝原因：

| 提示 | 拒绝原因 | 观测值与上游门槛 |
|---|---|---:|
| 后退 | foot sliding | 超过 12 mm |
| 左侧移 | root speed | 2.06 m/s > 2.0 m/s |
| 慢跑 | root speed | 6.13 m/s > 2.0 m/s |
| 下蹲 | joint speed | 15.69 rad/s > 15 rad/s |
| 双臂上举 | joint discontinuity | 1.70 rad/frame > 0.5 |
| 右腿前踢 | joint discontinuity | 1.16 rad/frame > 0.5 |

慢跑和两条 discontinuity 是明显的生成异常；左侧移与下蹲只略微越界，说明质量门也可能
对边界动作偏保守，适合以后做阈值敏感性分析。后退则是接触一致性问题。这个诊断不支持把
门槛整体放宽：那会让明显不连续的动作进入 tracker。值得注意的是，右侧移通过了质量门，
却在 tracker 中失败最多，进一步证明生成质量和闭环可跟踪性是两个不同的失败层级。

远端任务在 72 个 episode 全部写入后因汇总脚本的直接执行 import path 出错而标记失败。
输入文件已由冻结清单固定；修复导入后在本地重建 summary，未重跑生成或物理评测。

### 6.4 固定 reference 的事后修复实验

在预注册压力测试失败后，我们对同一批 9 条生成动作做了两轮诊断性修复；没有重新生成、
失败重抽或按单条提示调参。由于修复策略是在看到原压力测试失败后制定的，这一节属于事后
实验，不能替代 6.3 的预注册反证。

| 结果 | 原压力测试 | repair v1 | repair v2 |
|---|---:|---:|---:|
| 新 reference 通过质量门 | 3/9 | 6/9 | **6/9** |
| 质量门通过项的 tracker success | 81.94%（6 条） | 73.15%（9 条） | **75.93%（9 条）** |
| 12 条提示端到端 success | 40.97% | 54.86% | **56.94%** |

v1 使用接触期 root lock、统一慢放和三角平滑；它能把双臂上举与踢腿送过质量门，但分别
慢放到 19.80 秒和 16.96 秒，证明“通过门”不等于 reference 合理。v2 对检测到的关节跳变
做局部 smoothstep 修补，将慢放限制在 2 倍，并且不平滑 root translation；两条动作缩短为
7.28 秒与 6.84 秒。v2 相对 v1 的端到端均值再提高 2.08 个百分点，双臂上举三个 seed 的
success 为 75%、50%、100%。但踢腿仍为 0%、0%、0%，seed 1 的整体 success/completion
只有 72.22%/85.39%，未达到每个 seed 75%/90% 的冻结门槛。v2 的平均 Empjpe、foot slide
和 jerk 也略差于 v1。因此结论是：局部修复改善了覆盖率并定位出问题，但尚未形成可信的
扩展 demo；下一步应诊断高层生成链，而不是继续把异常 reference 慢放到过门。

### 6.5 生成阶段归因与分段对照

适配器审计首先澄清了接口边界：OMG 已直接输出 Unitree G1 的 `qpos_36`，本项目没有在
OMG 与 tracker 之间执行 SMPL→G1 retarget。桥接步骤只做维度与关节顺序检查、四元数符号
连续化，以及 30→50 Hz 重采样。我们在同一批冻结的 9 条 OMG 输出上分别测量原始 30 Hz
轨迹和桥接后的 50 Hz 轨迹；6 条被拒动作都已在原始输出中违反对应约束。因此这 6 个失败
属于高层 OMG reference，而不是适配器制造的误差。

其中 5/6 条拒绝动作的最大关节跳变位于第 59→60 帧。为检验 OMG 的 60 帧扩散分段是否是
可操作的原因，我们按其 condition-sequence 接口做了一次事后探索对照。两组使用同样的
9 条冻结提示文本、同一 50M ONNX 模型，每个“提示×方案”只生成一次，不允许重抽：

| 生成方案 | 质量门通过 | 最大跳变位于 59→60 |
|---|---:|---:|
| 单段 60 帧：`text: prompt` | **6/9** | 0/9 |
| 双段 120 帧：`text[2]: prompt` | **3/9** | 8/9 |

单段方案使后退、双臂上举和右腿前踢由拒绝变为通过；左侧移、慢跑和下蹲仍未通过，说明
分段边界是当前长动作的主要故障，但不是生成器的唯一局限。由于该对照是看到失败后的探索
实验、每格只有一次生成，而且 60 与 120 帧也改变了动作时长，不能把 6/9 对 3/9 当成
总体模型性能的统计结论。它足以决定工程顺序：先修复或避开多段拼接，并用多个生成随机种子
复验；仍在单段内失败的动作类别，再比较其他预训练生成器或做定向 Text2Motion 微调。这项
诊断没有给出重训 motion-tracking 小脑的理由，也尚不足以支持从零重训整个 OMG。

### 6.6 三生成种子短时动作闭环

按照 6.5 的工程顺序，我们先把固定 OMG 中已有的 diffusion continuation 接入离线生成，
并在慢跑、双臂上举和右腿前踢上使用生成种子 0、1、2。三种 continuation 设置都降低了部分
分段跳变，但质量门仍全部为 0/9；最强设置把接缝最大步幅均值从 1.91 降到 0.96 rad/源帧，
仍不足以过门，而且慢跑在第一段内部已经存在异常根速度。因此现有 continuation 不能作为
可信的长动作修复。

随后对全部 9 个新提示做三生成种子的配对单段/双段对照。每个“提示×生成种子”中的第一段
完全相同，每格只生成一次且不重抽：

| 方案 | 质量门通过 | 最大跳变在 59→60 |
|---|---:|---:|
| 单段 60 帧 | **21/27（77.8%）** | 0/27 |
| 双段 120 帧 | **9/27（33.3%）** | 22/27 |

配对结果为：两者均通过 9、仅单段通过 12、仅双段通过 0、两者均失败 6。我们把全部 21 条
单段通过项送入三个冻结小脑，每条做 4 次 observation-noise repeat；另外 6 条继续按端到端
失败计数。由于 60 个 30 Hz 帧按官方时钟重采样后为 99 帧，而上游 loader 默认过滤少于
100 帧的 reference，评测器增加了显式 `--min-ref-frames 2`。这一覆盖只取消加载过滤，不
修改仿真终止条件、策略或指标。

| 指标 | tracker seed 0 | tracker seed 1 | tracker seed 2 | mean ± sample SD |
|---|---:|---:|---:|---:|
| 21 条通过项 tracking success | 82.14% | 80.95% | 78.57% | **80.56% ± 1.82%** |
| completion | 94.62% | 94.56% | 93.74% | **94.31% ± 0.49%** |
| 27 单元端到端 success | 63.89% | 62.96% | 61.11% | **62.65% ± 1.41%** |

三个 tracker 都通过 75% tracking、90% completion 和 60% 总体端到端冻结门槛。因此可以
新增“事后短时动作、多生成种子 demo”结论。但它对生成种子仍敏感：生成 seed 0 在每个
tracker 下的端到端结果都低于 60%。鞠躬、后退、挥手和双臂上举在其通过的 generation 上
全部成功；踢腿只有 12.5%，左右侧移为 45.8%/50.0%。故大脑侧下一步是带显式速度/过渡
连续性目标的长动作微调或替换；若 demo 必须包含踢腿和侧移，小脑也需要针对这些 reference
做定向适配，而不是全量从零重训。

### 6.7 长动作修复、闭环与域适配

为避免把短动作结果外推到双段动作，我们固定同一 27 个“提示×生成种子”双段输出，不重抽，
按预注册门槛逐级测试。直接把第二段的姿态和速度残差做 C1 衰减，质量通过率从 9/27 提高到
14/27，但损失了原来 9 条通过项中的 1 条。第二版把水平平移和 yaw 视为刚体平面坐标变换，
只对超阈值关节通道做选择性 C1 衰减，保住 9/9 原通过项并达到 15/27。最后一个明确标为
adapter sanitizer 的阶段，统一处理脚滑、悬空和内部关节速度异常，但拒绝用时间拉伸“修复”
超速，最终达到 18/27，三个生成种子恰好都是 6/9。它恢复了 2 个脚滑和 1 个关节速度失败，
仍留下 7 个超速和 2 个悬空。这个 18/27 只能解释为接口清洗后的覆盖率，不能替代原始 OMG
双段质量 9/27。

全部 18 条通过项进入三个冻结小脑，每条做 4 次 observation-noise repeat，共 216 episodes：

| 指标 | tracker seed 0 | tracker seed 1 | tracker seed 2 | mean ± sample SD |
|---|---:|---:|---:|---:|
| tracking success | 65.28% | 72.22% | 73.61% | **70.37% ± 4.46%** |
| completion | 84.09% | 87.04% | 87.25% | **86.13% ± 1.76%** |
| 27 单元端到端 success | 43.52% | 48.15% | 49.07% | **46.91% ± 2.98%** |

三套小脑均未同时达到 75% tracking、90% completion，端到端也都低于 60%。失败高度集中：
鞠躬和右手挥手为 100%，后退为 97.22%，双臂上举为 80.56%；踢腿为 0%，右侧移为
20.83%，下蹲为 16.67%。因此“质量门通过”不是“长时闭环可执行”的充分条件。

随后只对 tracker seed 0 做两个有留出的域适配预检：生成 seed 0/1 的 12 条动作用于训练，
seed 2 的 6 条动作完全留出。纯 12 条长动作追加 300 iterations 后，留出 success 仅从
58.33% 升到 62.50%，而 60 条原生动作 success 从 91.67% 降到 73.33%。加入 120 条确定性
原生 replay 后，留出 success 反而降到 50.00%，原生 success 为 76.25%。两次都未通过
“留出提升 + 原生保护”门槛，故没有把微调扩展到其余两套小脑。这否定的是当前小样本配方；
未来若继续，应使用更大的混合长动作训练集、较弱或分层更新，以及动作语义留出，而不是在
12 条 reference 上继续增加迭代。

## 7. Demo 与视觉解释

seed 0 是首选展示 checkpoint，并已为三条 OMG 提示生成并排视频。左侧是 OMG 目标骨架，
右侧是 MuJoCo 闭环 G1。远端环境没有可用的 headless OpenGL context，因此视频采用二维
运动学投影；物理 rollout 本身仍使用 MuJoCo 和硬件 observation-noise 模型。

提交目录的 `assets/demo/` 中包含向前走、左转和右转三段精选视频及缩略图，可以直接在
GitHub 上查看。它们只用于直观展示，不替代定量评测。

右侧不应因为“看起来更抖”就被直接判定为成功或失败。成功由 episode evaluator 的完成、
偏差和终止条件决定；视觉抖动也可能来自闭环纠错、投影、关节结构和噪声。最终结论依赖统一
协议下的 success/completion/MPJPE/foot-slide/jerk，而不是只挑一段视频作主观判断。

## 8. 失败路线及其作用

项目早期研究过 residual Motion Cerebellum、preview 与不同平滑/纠偏版本。这些实验帮助
确认了一个重要边界：如果原始 tracker 在自身分布上都没有充分训练，外挂 residual 的收益
难以解释；而过强修正还可能让成功指标与视觉质量分离。主线因此回到原仓库优先：修复数据
加载问题、使用正式 rollout 规模、从零训练，并只在 tracker 可信后测试 Text2Motion 接口。

另一个关键诊断是“500 clip 限制”并非内存或数据损坏，而是旧 checkpoint 保存的 500 个
adaptive-sampling weights 与扩大后的采样池长度不一致。修复为长度不匹配时丢弃旧权重并从
uniform sampling 重启后，750、1000、1028 clip smoke 均通过。该修复服务于 warm-start
诊断；最终三种子 clean-start 不依赖旧 checkpoint。

## 9. 结论、边界与下一步

本项目完成了预定 demo：预训练文本动作生成器与独立训练的运动小脑通过明确接口连接，并在
带噪 MuJoCo 闭环中稳定运行。三个 clean-start 训练种子的原生成分布成功率均约 91%，原三
条文本动作总体成功率为 34/36 episodes。扩展压力测试同时给出必要的反证：在不挑选提示的
协议下，完整系统只有 40.97% 端到端成功率。固定 reference 的事后修复可把该均值提高到
56.94%，但仍未通过全部种子门槛，不能升级原压力测试结论。事后的三生成种子短时方案则以
21/27 的质量通过率和 62.65% 的平均端到端成功率通过其冻结门槛，说明可可靠展示约 2 秒的
单段动作，但不能外推到双段长动作。双段长动作即使经清洗达到 18/27 质量门，三个小脑的
平均 tracking/end-to-end 仍只有 70.37%/46.91%，小数据微调也未通过原生回归保护。结果
支持以下表述：

> 一个可复现的 Text2Motion→Motion Cerebellum 演示、三种子正式规模 G1 跟踪基线，以及
> 一个明确限定为短时动作的多生成种子扩展 demo。

不能据此声称：真实 G1 已部署成功、任意开放文本都能稳定执行、foot sliding/jerk 在所有
定义下都改善 10%，或严格复现了上游论文全部数据和指标。当前最有价值的后续实验是扩大
文本提示和动作类型并按语义类别分层报告。现有 continuation 和后处理均不足，因此长动作应
进入带过渡/速度目标的生成器定向微调或替换；踢腿和侧移若属于目标 demo，小脑侧需要更大
的混合长动作语料与语义留出，不能复用本次 12 条小样本配方。真实机器人实验需要独立的安全审查、sim-to-real
配置和硬件授权，不属于本报告范围。

## 10. 复现与证据索引

- 技术过程与全部中间结果：`projects/text2motion_cerebellum/README.md`
- 提交版主结果：`projects/text2motion_cerebellum/results/main_results.json`
- 提交版扩展提示结果：`projects/text2motion_cerebellum/results/expanded_prompt_results.json`
- 提交版质量门诊断：`projects/text2motion_cerebellum/results/prompt_quality_diagnostics.json`
- 事后固定 reference 修复结果：
  `projects/text2motion_cerebellum/results/reference_repair_results.json`
- 生成阶段归因与分段对照：
  `projects/text2motion_cerebellum/results/generator_diagnosis_results.json`
- 多生成种子短时闭环结果：
  `projects/text2motion_cerebellum/results/short_horizon_results.json`
- 长动作修复、三小脑闭环与域适配负面结果：
  `projects/text2motion_cerebellum/results/long_horizon_results.json`
- 外部数据、checkpoint、episode 级数据与视频的提交边界：
  `projects/text2motion_cerebellum/DATA_AND_LICENSES.md`
- 完整 episode 数据、冻结 manifest 和展示视频保留在 Git 忽略的
  `outputs/remote_text2motion_mainline/` 下，不随仓库提交。

上游资料：

1. `suning-git/motion_tracking`：<https://github.com/suning-git/motion_tracking>
2. 原仓库中文教程：<https://github.com/suning-git/motion_tracking/blob/main/docs/TUTORIAL_CN.md>
3. OMG 项目页：<https://tsinghua-mars-lab.github.io/OMG/>
