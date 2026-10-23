# nano_multimodal — 三个模态,一套自回归训练

三个模型:**文本语言模型**、**文本→动作**、**视频世界模型**,外加文本线上的
**compute-optimal scaling law**(`scaling.py`,5 档模型 × 500M token,两张卡 38 分钟)。
tokenizer 和编码好的数据都是给定的,所以三条线在训练侧是同一件事——整数序列进、
整数序列出的自回归模型。两个 web 面板让这件事**看得见**:训练时喂进去的是什么,
训出来的又是什么。

要读代码,从 `spec.py` → `train.py` → `configs/` 三份 yaml 开始。

## 这三个模型跑出来是什么样

单张 RTX 5090,三条线各一次完整训练:

| 线 | 模型 | 步数 | 训练时长 | 监督 token | 最好 val CE |
|---|---|---|---|---|---|
| 文本(先跑通) | d6 / 35.8M | 3,000 | ~15 分钟 | 393M | 4.1561 |
| 文本(冠军) | d12 / 135.3M | 20,640 | 4.6 小时 | 2.705B | **3.4113** |
| 动作 | d6 / 36.2M | 16,000 | 8.7 分钟 | 131M | 2.3685 |
| 视频 | d12 / 183.3M | 9,059 | 62 分钟 | 377M | 5.2476(起点 11.07) |

视频这一行训练时,视频行里的动作 token 被标成了视频类型(类型嵌入用错,现已修正,见 `sources.py`),
重训后数字会略有不同。

本页的表(包括下面 scaling 一节)都是 `head_ce=liger` 时测的。2026-10-22 起配置改成 `compiled`:
单张 RTX 5090 上训 300 步对照,每步文本 d12 805 → 630 ms、视频 d12 425 → 272 ms、动作 d6 32 → 12 ms,
训练时长会相应缩短。读数:每种实现各跑一次 300 步,第 299 步 val 差 0.001–0.011;视频两种实现各跑了两次,
同一种实现两次之间就差 0.004–0.007,没量出稳定的实现差异。

参数量是**整个系统**(trunk + LM head),也就是训练器启动时打印的那个数。同样是
d12/768,三条线的参数量不同 —— 差的全在 un-embedding 上,词表越宽它越大
(文本 32,768 / 动作 33,280 / 视频 64,037)。只数 trunk 会得到另一套数(d12/768
都是 110M),两套数不要混进同一张表。

文本冠军的步数不是手填的:`max_steps: -1` 让 core 按参数量算 Chinchilla 预算
(135M × 20 = 2.705B token),再除以每步 token 数。它的最好 val **就是最后一步** ——
喂满 Chinchilla 预算都还没到转折点。

注意三点。一,当时 core 在自动步数下不存最后一步(core 的 bug,2026-10 查出),所以存下来、
网页面板加载的是第 20000 步的存点(val 3.4205);表里的 3.4113 是第 20639 步的读数,那个
模型没有存下来。二,动作线的验证集随 core 换了一批:动作数据源(`modalities/motion` 的 `T2MDataSource`)2026-10
改用 core 的 ResumableDistributedSampler 抽样,评估冻结的前 160 对验证样本因此换成了另一批,和原来
只有 1 对相同。同样的权重在新的 160 对上读数低 0.10–0.16(d6 第 15999 步的存点:原来那批上
2.3649,新的一批上 2.2095,都是 2026-10 重评的),所以表里和下面那段里的动作数,不能和用新
代码训出来的读数直接比。
三,这些数都是在 bf16 存储(参数、梯度、AdamW 动量都按 bf16 存)下量的,那是
core 当时唯一的存储方式(v0.2.0 及以前)。存储方式现在是配置项 `param_dtype`(继承自 core 的
`train_base.yaml`),命令行写 `param_dtype=bf16` 就回到当时的存储方式,但不会逐位复现当时
的数(之后 core 还有别的改动)。

动作线上把模型放大到 d12(136.1M,是 d6 的 3.8 倍),同样的语料同样的配方,它在第 5,500 步就到
val 2.3811,然后一路漂到 22,000 步的 2.6943,过拟合了;d6 到 16,000 步还在降(2.384 → 2.3685)。
原因可量:Bones-SEED 一个 epoch 是 450,594 行 = 30.4M 监督动作 token,
而一个 36.2M 的模型按 Chinchilla 想要 ~720M(约 24 遍)。**语料是上限,不是模型。**
推理面板的 checkpoint 选择器把 d12 的「谷底」和「末尾」都标了出来,可以切着看。

★ **同样的 12 层 / 768 宽,在文本上不过拟合、在动作上过拟合** —— 表里文本冠军那行和上面这段
放在一起就是证据。
所以"过拟合"不是模型的属性,是**模型容量与语料量之比**的属性。

数据体积:视频 token 子集 737 MB、动作 cache 113 MB、文本 tokenizer 0.7 MB;
解码用的 Cosmos decoder 120 MB、动作 codec 70 MB、SMPL 静止骨架 1.9 KB。

### 文本线的 scaling law

五档模型(depth 2/3/4/6/8,dim = 64×depth)各喂 **500M token**,恒定学习率,
两张 5090 跑 38 分钟。前沿指数 **a = 0.446**(N_opt ∝ C^a),包络上有 4 处曲线交叉。

| depth | 非嵌入参数 | 最终 val CE |
|---|---|---|
| 2 | 0.4M | 5.1678 |
| 3 | 1.3M | 4.8097 |
| 4 | 3.1M | 4.6152 |
| 6 | 10.6M | 4.3726 |
| 8 | 25.2M | 4.2065 |

★ **这个数字不能脱离预算引用。** 前沿指数随覆盖的算力增长,因为包络的高算力端还在填:
把同一批曲线截断后重拟,100M 处是 0.566、300M 是 0.468、500M 是 0.446;
`exemplars/text_pretrain` 的 2B 研究（7 月那批 bf16 曲线）在 1B 处 0.507、2B 处 0.519。
**a ≈ 0.45 @ 500M 不等于"Chinchilla 0.5 已复现"**——`scaling.py fit` 因此总是把预算和
数字一起打印。同口径对照下,本项目在 300M 处与那份参照实现完全一致(两边都是 0.468)。

## 三条线的全部差别

```
                  ①激活带              ②数据源            ③一行怎么摆        ④哪些位置算 loss
文本    text, control              TextDataSource     bos text… eos          全部
动作    text, control, motion      T2MDataSource      caption + 码           动作半边
视频    control, video, action     VideoRowSource     帧码与动作交错          被预测的 4 帧

以下三条线完全相同:1D RoPE · next-token CE · GPT+LMHead · MixedDataLoader · core Trainer
```

`train.py` 里没有一处 `if line == ...`。把三份 config diff 一下就看得见为什么。

## 数据从哪来 —— 你自己下载、自己处理

**这个仓库不发数据。**

下面标 ✔ 的命令在 RTX 5090 上真跑过(冒烟规模);标 · 的只核对过入口和参数,没有实跑
——它们是下载动作,或者依赖下载的产物。时间标注同理:实测的写"实测",外推的写"外推"。

### 文本 —— 全自助,不需要任何账号

```bash
·  python -m exemplars.text_pretrain.data.download_shards  # FineWeb sample-10BT → outputs/base_data/
✔  python -m modalities.text.train_tokenizer               # 训你自己的 BPE(vocab 必须仍是 32768)
✔  python -m projects.nano_multimodal.train --config-name text
```

`train_tokenizer` 不带 `--out` 会**写进默认位置 `outputs/tokenizer/`,原地覆盖**。
想试跑就加 `--out <别处>`;`get_tokenizer` 读的是 `NANOINFRA_TOKENIZER_DIR`,指过去即可。

不带参数会下**六个 shard(约 13 GB)**,因为 val 是**按文件名钉死的**(`text.yaml` 指定
`shard_005`),train 是"其余全部"——少下一个,训练语料的定义就变了。
d6 教学跑 3,000 步只吃 0.39B token,一个 shard 够用;复现 d12 冠军点(2.705B)要四个。

### 视频 —— 全自助,不需要任何账号

```bash
python -m exemplars.nano_world_model.data.download --shards 2   # 公开 VizDoom + Cosmos codec
python -m exemplars.nano_world_model.data.encode                #   像素 → 码
python -m exemplars.nano_world_model.build_cache                #   码 → 定步长 memmap
python -m projects.nano_multimodal.train --config-name video \
    video_cache=exemplars/nano_world_model/outputs/cache/dv128_17f
```

视频 codec(Cosmos DV4×8×8)是**冻结的公开权重,不用训**——`download.py` 会一起取。
最后一行是本项目的训练器**直接吃样板产出的 cache**:两边的形状契约逐项相同
(17 帧 / 128px / 每帧 256 码 / 5 个 latent 帧 / td=4),只是行数不同。
**已实测**:拿样板那份 4,733 行的 cache 训我们的视频线,起始 loss 恰好
`ln(64037) = 11.067`,30 步降到 10.46,MFU 47%。

### 动作 —— 骨架免费,**标注在牌照墙后**

```bash
·  python -m exemplars.nano_motion.data.download           # LAFAN1(Ubisoft 动捕,免账号)
✔  python -m exemplars.nano_motion.data.download --check   #   随时看盘上有什么
✔  python -m exemplars.nano_motion.data.prepare            #   → rot139 特征(幂等,已存在就跳过)
✔  python -m exemplars.nano_motion.train_codec             #   30,000 步 ≈ 16 分钟(外推)
✔  python -m exemplars.nano_motion.data.encode --codec <上一步的 .pt>
```

**这条链路给你的是 tokenizer 和一个无条件动作模型,不是文本→动作。** LAFAN1 没有 caption。

要训本项目的**文本→动作**线,得手工下载两份许可数据:
[AMASS](https://amass.is.tue.mpg.de)(逐个子集注册、接受许可)和
[HumanML3D](https://github.com/EricGuo5513/HumanML3D)(caption)。脚本抓不了,这是设计如此。
下好之后 `prepare.py --source amass`,后面照旧,产出的 `t2m_*.npz` 就是 `motion.yaml` 要的。

**⚠ `encode.py` 写进共享的 `outputs/motion_caches/`,文件名由数据源推导,`--limit` 不改名字。**
所以一次 `--limit` 冒烟会**原地顶掉**同名的真 artifact(我们踩过)。好在它写的 sidecar
如实记着是哪个 codec 编的,`assembly.py` 会读它并与 `spec.MOTION_CODEC` 对账。

### 三条线在"tokenizer 从哪来"上恰好是三种答案

| 线 | tokenizer | 代价 |
|---|---|---|
| 文本 | **自己训一个** BPE | 秒级 |
| 动作 | 自己训一个 codec,配方在 `modalities/motion/tokenizers/rot139_kin_fsq2/recipe.yaml` | 1,000 步实测 33 秒 → 30,000 步外推 16 分钟 |
| 视频 | **下一个别人训好的**(Cosmos,冻结) | 下载 |

第三种要注意:你**继承了它的天花板**。视频线的重建上界是量得出来的,模型再好也过不去。

## 跑

```bash
# 0. 文本 tokenizer(几秒),它给整个共享词表定尺寸
#    数据包里已经带了一份;要自己训就跑这行(vocab 必须仍是 32768)
python -m modalities.text.train_tokenizer

# 1. 训练:三条线,同一个编排器
python -m projects.nano_multimodal.train --config-name motion   # 分钟级
python -m projects.nano_multimodal.train --config-name video    # ~1 小时(单 A100)
python -m projects.nano_multimodal.train --config-name text     # ~4 小时

# 两张卡:launcher 决定卡数(RANK 只有 torchrun 会设),parallel 决定摆法
torchrun --nproc_per_node=2 --standalone \
    -m projects.nano_multimodal.train --config-name video parallel=ddp

# 2. 文本的 compute-optimal scaling law(ladder 天然可拆,一张卡一档)
python -m projects.nano_multimodal.scaling run --depths 8 6
python -m projects.nano_multimodal.scaling fit

# 3. 数据浏览器 + 推理 web(同一个进程、同一个端口)
python -m projects.nano_multimodal.serve.app --port 8800 --device cuda:1
```

`--device` 默认指向第二张卡:web 常驻三个模型加三个解码器,和训练挤同一张卡会 OOM,
而且报错不会告诉你原因。

## 状态

**六步全部完成,端到端可跑。** 起始 loss 三条线全部等于 `ln(vocab)`;视频行与
`exemplars/nano_world_model` 的实现逐位一致(`tests/test_video_row.py`);2 卡 DDP 与
续跑均已验证。三个解码器目检通过;三档采样器 greedy 逐 token 一致,CUDA graphs 3.5×
(0.25 s/帧,交互 rollout 4 fps);数据浏览器和推理 web 三条线都跑通,
`/api/row` 与训练 loader 逐 id 相等。

**还没做的,说清楚**:本项目的 scaling 阶梯与 `exemplars/text_pretrain` 的曲线之间有
一处未解释的系统偏差(等预算下本项目的 val 低 0.03–0.10 nat,且随模型增大)。已排除
val 划分、分隔符占比、打包方式;最可能是两份曲线不在同一个词表下,未验证。
**在弄清楚之前,别把两条曲线画在同一张图上。**

`projects/` 是有意的落点——`exemplars/` 的门槛是"跑通过、数字钉过、别人照做能复现";
三条都满足了再搬。

项目**写**出来的东西(checkpoint、视频子集、scaling 曲线、目判画廊)全部落在
`outputs/`;项目根目录只放**发**出去的东西。这条规则由 `spec.py` 里的路径执行,不靠人记,
而 `projects/*/outputs/` 正是本仓库 `.gitignore` 已经管住的地方 —— 一个装着几十 GB 权重的
目录,名字必须是 git 早就被告知别碰的那个。

`outputs/` 不进版本库,所以这一份 README 是自包含的:上面那张表就是全部实测数字。
