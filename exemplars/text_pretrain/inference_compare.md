# Inference across the budget ladder — one architecture, four budgets

Same architecture (the spec.py d12): decoder-only transformer, **12 layers,
dim 768, 6 heads** (head_dim 128), RoPE, vocab 32768 (BPE), trained at
sequence_len 512, **135.3M total params**. Every rung is a PROPERLY SCHEDULED
run for its budget (full warmup + warmdown — a budget-B model, not a mid-run
read of a longer run):

| budget | wall time | val loss, **bits/byte** | (nats/token) |
|---|---|---|---|
| **20M tokens** | ~2 min, 2×5090 | **1.759** | 5.431 |
| **131M tokens** | ~7 min, 2×5090 | **1.411** | 4.356 |
| **1.1B tokens** | 64.7 min, 2×5090 | **1.251** | 3.861 |
| **2.7B tokens** (Chinchilla 20 tok/param; the pinned champion) | ≈4.4 h, 1×5090 | **1.236** | 3.817 |

All four are direct measurements on the SHIPPED checkpoints over the common
2 M val window (one protocol, comparable numbers). Each run's own training log
reads slightly lower — different eval window and/or the final warmdown tail
(e.g. the champion's final-step eval: CE 3.8101).

Anchors for bits/byte: random bytes = 8; gzip on English ≈ 2.5–3. Even the
2-minute rung beats gzip; the champion compresses English to 1.23.

*Metric note*: bits/byte uses the tokenizer's true byte table (this eval
window: 4.456 bytes/token). `token_bytes.pt` is now INSTALLED, so freshly
logged `val/bpb` is true bits/byte; HISTORIC training logs' "bpb" column was
bits-per-token (the byte table used to fall back silently to ones).

Sampling: same prompts, same seed (temp 0.8, top-k 40), core KV-cache engine.
All checkpoints are SELF-DESCRIBING; each loads with one `load_system(ckpt_dir)`.
**Bold = prompt, plain = the model's continuation.**

Reproduce (each rung = the same spec with a max_steps override):
```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun --nproc_per_node=2 --standalone \
  -m modalities.text.train_text model.depth=12 max_steps=<150|1000|8400> \
  use_compile=true checkpoint.enabled=true checkpoint.save_dir=<dir> \
  checkpoint.save_every=<150|500|2800> checkpoint.keep_last_n=1
# 2.7B rung: pretrain.py (stage 1). Generation: inference.py's prompts/seed
# via load_system(<ckpt>, sequence_len=256) per checkpoint.
```

## 20M tokens (~2 min) — bpb 1.759

> **The history of the Roman Empire**, and the world of the North American Society. It seemed to be the right to be one of our modern-day.
The new trend has been a popular approach. The world is the most respected the world of a cultural tradition and a modern world around the world of the culture. There is a way of the world and the world that has been the

> **In a surprising turn of events, scientists have discovered** in several times, more often than 20 years, and they can be seen without any idea to get the ability to come to the world by the end of the month. That is about why it's not that if you like a big problem that you'll end. This includes:
- For example, you know, you can see that on the

> **The best way to learn a new language is** the perfect atmosphere, but there’s a lot of the design. The results show is also a fantastic way to be a lot of the real-life experience. It’s important to be a way to play with your work, and it is a beautiful place. We like all have the ones to get the most important to use the design of the way to

> **Once upon a time, in a small village** in the village, it seemed to be a good thing. He says to find out this game has turned a new one on the side of the world and its best way to the most expensive and the time is available to the game. In her past we are working for the players will be working out on the event. With the best and what we

## 131M tokens (~7 min) — bpb 1.411

> **The history of the Roman Empire**, and the history of the ancient tomb of the Roman Empire. This story is an excellent story for our modern life.
The new year has been a year since its inception.
The year began with the introduction of a new generation and a sense of wonder that one of the most important aspects of the ancient Romeo period. The early years of creation,

> **In a surprising turn of events, scientists have discovered** that "the most dangerous objects in the world are at risk," says an American scientist, "the worst-case scenario for that person."
But the first thing to do is "profiler" a disaster that has always been a disaster, because a fire is a disaster that caused a major storm.
The current storm has been known as a meteor

> **The best way to learn a new language is** to know what we will be doing. This is the key to a learning process or a practice that will help you become a part of your life.
To become curious about how an attitude is to take you through the world, it is essential to learn a new language in your own vocabulary. It is also important to learn the language of your words that

> **Once upon a time, in a small village** in the village of Fijam, New Orleans, the village of Fijam, is called the village of Havana. The river is a UNESCO World Heritage Site, by which the city of Havana was built, in its possession, which is the oldest of the town in the village of Fijam. With its proximity to the town

## 1.1B tokens (64.7 min) — bpb 1.251

> **The history of the Roman Empire**, and the history of the Roman Empire, is an absolute revelation. The first and greatest military conflict was the Roman Empire, the Battle of the Roman Empire. Both that were part of the Roman Empire, the Civil War, and then that was the Battle of the Roman Empire. There was a certain amount of confusion about how to get to know the

> **In a surprising turn of events, scientists have discovered** that most of the bacteria present in the digestive tract are beneficial in treating anemia and other symptoms of a chronic condition. With that in mind, the researchers suggest that a large number of pathogens can play a role in our symptoms.
- 2.2. Prevention and Treatment of Digestive Hypertension
- 3.2. Prevention and

> **The best way to learn a new language is** to know what we are talking about. This is the key to a great language or a language that makes you learn a lot. Learn how to learn a new language, find an unfamiliar language, and learn your language better.
The more experienced you learn a language, the more excited you become. This is the key to a great language immersion.

> **Once upon a time, in a small village** in the woods, at the beginning of the seventeenth century, the king of Humboldt, a son of the same name, came by and ate a bread of the same name and put on his coat to the king. In the years immediately thereafter, in the seventeenth century, the king of Humboldt became king of H

## 2.7B tokens (the champion) — bpb 1.236

> **The history of the Roman Empire**, and the birthplace of the Roman Empire. It was founded by the monarchy of the 3rd century B.C. in 1879, built in 1883 by Sir William S. Everson. There are some 260 miles of paved road, 110 miles of public transit and 14 miles of cycle trails

> **In a surprising turn of events, scientists have discovered** that "the most dangerous secret lies in the amount of nutrients they contain that will help us to function better than we are now."
Dr. Robert L. Williams
Dr. Robert Williams is the director of the National Institute of Kidney and Blood Disorders (NIDCD) and the author of the book "The Secret To Kidney Stones." His

> **The best way to learn a new language is** to know what languages you speak, what people say on the internet, what products or services you have on the Internet, a language you have never heard before, and a language that you know is your own language, just as a language that you can learn it in the context of your native language. You may also learn a few words as you speak

> **Once upon a time, in a small village** in the south of England where I grew up. I was a good boy and a good boy, but I could not think of a country better than England. I was born in 1884 and spent most of my adult life in the south west of Scotland, south of England, and north of Scotland. I still live here today and have lived

---

**What the ladder reads like.** The differences are not in grammar (even the
2-minute rung is locally fluent); they are in **prompt-following and
coherence**: 1.759 ignores the prompt and drifts (Roman Empire → "the North
American Society"); 1.411 holds the topic but confabulates hard ("the ancient
Romeo period"); 1.251/1.236 write structured, self-consistent fiction. Roughly
every −0.2 bits/byte buys a visible capability step.

**And where the eye stops working**: the last two rungs (1.251 vs 1.236, a
0.015 gap) are hard to tell apart in samples — large gaps are visible, small
gaps only the loss metric resolves. That is why the judge is a curve, not an
impression.

Checkpoints: `models/exemplars/text_pretrain_d12_lr3e-4_{20M,131M,1h}/` and the
champion `models/exemplars/text_pretrain_d12_lr3e-4/step_020000`.
