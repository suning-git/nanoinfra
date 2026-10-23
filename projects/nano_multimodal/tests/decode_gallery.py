"""
decode_gallery.py — render one real training row per line, for a human to look at.

The automated checks in test_decode.py prove the decoders are self-consistent. They
cannot prove the pictures are RIGHT: a skeleton with a swapped axis, a frame decoded
in the wrong channel order, a caption paired with the wrong clip all pass every
assertion and are obvious to the eye in one second. So this writes a page and asks
for one second of eye.

    python -m projects.nano_multimodal.tests.decode_gallery

Writes outputs/gallery/decode_gallery.md plus its images.
"""

import numpy as np

from projects.nano_multimodal import assembly, decode, spec

OUT = spec.GALLERY_DIR


def save_gif(frames, path, fps=10):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def save_strip(frames, path, n=6):
    """A contact sheet: n evenly spaced frames side by side, so the page reads
    without playing anything."""
    from PIL import Image
    picks = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    tiles = [Image.fromarray(frames[i]) for i in picks]
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * w, 0))
    sheet.save(path)
    return list(picks)


def row_of(line, device="cuda"):
    config = assembly.load_config(line)
    vocab = assembly.assemble_vocab(line, config)
    loader = assembly.build_loader(line, config, vocab, "val", device, batch_size=1)
    batch = next(iter(loader))
    return vocab, batch["idx"][0].cpu().numpy().tolist(), batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# decode/ 目检画廊",
          "",
          "**要看的是**:三条线的解码器有没有把整数还原成看得懂的东西。自动测试只能证明",
          "自洽(units 能拼回字符串、8x8 块正好铺满一帧),证明不了**画得对**——骨架轴向搞反、",
          "帧的通道顺序错了、caption 配错了片段,这些都能通过所有断言,而肉眼一秒就看出来。",
          "",
          "每一行都是从**训练用的那个 loader** 里取出来的真实 val 行,不是另开一条路读的数据。",
          ""]

    # --- text ---------------------------------------------------------------
    vocab, row, _ = row_of("text", device="cpu")
    r = decode.render("text", row, vocab)
    units = [u for u in r.units if u["band"] == "text"][:40]
    md += ["## 文本", "",
           f"行长 {len(row)}(= sequence_len;样本有 sequence_len + 1 个 token,位置 i 预测 i+1),"
           f"其中 {len([u for u in r.units if u['band']=='text'])} 个文本 token。", "",
           "**分段**:`" + " | ".join(f"{s['label']}×{s['end']-s['start']}" for s in r.spans) + "`", "",
           "**前 40 个 token 的切分**(`|` 是 token 边界,这是 BPE 真正做的事):", "",
           "```", "".join(f"{u['preview']}|" for u in units), "```", "",
           "**解码全文开头**:", "", "> " + r.media[:400].replace("\n", " "), "",
           "> 注意行内出现的第二个 `bos`:打包的文本流在**文档边界**插 bos,"
           "所以一行里可能跨越两篇文档。", ""]

    # --- motion -------------------------------------------------------------
    vocab, row, _ = row_of("motion", device="cpu")
    r = decode.render("motion", row, vocab, device="cpu", size=160)
    save_gif(r.media, OUT / "motion.gif")
    picks = save_strip(r.media, OUT / "motion_strip.png")
    n_codes = len([u for u in r.units if u["band"] == "motion"])
    md += ["## 动作", "",
           "**分段**:`" + " | ".join(f"{s['label']}×{s['end']-s['start']}" for s in r.spans) + "`", "",
           f"- {r.notes[1]}", f"- {r.notes[0]}", "",
           f"**要看**:骨架是不是站着的人(不是躺着、不是倒着);{n_codes} 个 code 的动作"
           "和 caption 对不对得上;有没有脚在地上滑。", "",
           f"![strip](motion_strip.png)", "",
           f"(上面是第 {list(picks)} 帧;下面是完整动画)", "",
           "![motion](motion.gif)", ""]

    # --- video --------------------------------------------------------------
    vocab, row, _ = row_of("video", device="cuda")
    r = decode.render("video", row, vocab, device="cuda")
    save_gif(r.media, OUT / "video.gif", fps=8)
    picks = save_strip(r.media, OUT / "video_strip.png", n=6)

    # One integer -> one 8x8 patch, drawn.
    vids = [u for u in r.units if u["band"] == "video"]
    pick = vids[len(vids) // 2 + 37]
    frame = r.media[min(pick["frame"] * spec.CODEC_TEMPORAL_DS, len(r.media) - 1)].copy()
    x, y, w, h = pick["box"]
    frame[y:y + h, x:x + 1] = frame[y:y + h, x + w - 1:x + w] = [255, 0, 0]
    frame[y:y + 1, x:x + w] = frame[y + h - 1:y + h, x:x + w] = [255, 0, 0]
    from PIL import Image
    Image.fromarray(frame).resize((384, 384), Image.NEAREST).save(OUT / "video_patch.png")

    md += ["## 视频世界模型", "",
           "**分段**:`" + " | ".join(f"{s['label']}×{s['end']-s['start']}" for s in r.spans) + "`", "",
           f"- {r.notes[0]}", f"- {r.notes[1]}", "",
           "**要看**:画面是不是 Doom 的走廊(颜色不对就是通道顺序错了);"
           "帧与帧之间是不是连贯;按键序列和画面里的转向/前进对不对得上。", "",
           "![strip](video_strip.png)", "", "![video](video.gif)", "",
           "### 一个整数 = 一个 8×8 块", "",
           f"token 位置 {pick['tokens'][0]},码 {pick['code']},第 {pick['frame']} 个 latent frame,"
           f"码网格 ({pick['grid'][0]},{pick['grid'][1]}) —— 红框就是它负责的那 8×8 像素:", "",
           "![patch](video_patch.png)", "",
           "这一张是数据浏览器里最难用嘴讲清的那个交互的静态版。", ""]

    (OUT / "decode_gallery.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT/'decode_gallery.md'}")


if __name__ == "__main__":
    main()
