# tests

一步一个验收物。每一行的第三列说的是**为什么是它**——测试要挡的是"错了不报错"
的那一类,不是"跑不起来"的那一类:

| 步 | 测什么 | 为什么是它 |
|---|---|---|
| 1 | 三条线的装配:band 偏移、vocab_size、n_token_types=6 | 偏移错一位照样训得好好的,只是解码出乱码 |
| 2 | `VideoRowSource` 拼出的行与 cache 逐位一致;`supervision_mask` 的 0/1 图案与 `[256 码 + 4 动作]` 周期严格对齐 | 最容易错、最难发现的一处:错了不报错,只是学错东西 |
| 4 | `decode/` round-trip:已知 clip → 编码 → 解码 → 比对 | 渲染器错了,后面所有目判都在评价渲染器 |
| 5 | 快路等价:teacher-forced argmax 与 greedy 对上参考实现 | 又快又微妙地错,比慢更糟。先证正确,再报速度 |

多卡的正确性不在这里重测:`core.parallel.NanoDDP` 由
`exemplars/nano_world_model/tests/test_ddp.py` 覆盖,本项目经同一个
`Trainer(ddp=...)` 接缝使用它。
