# -*- coding: utf-8 -*-
"""
辅助脚本（不修改 hw5_dpo.py）：按阶段逐题打印模型输出。
供实验报告 4.3 逐题输出记录、4.4 人工检查、6.1 失败案例使用。

用法（在 hw5_dpo 目录下）：
    py hw5_perq.py
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import PeftModel

import hw5_dpo as H

STAGES = [("基线", None)]
if os.path.exists(H.SFT_DIR):
    STAGES.append(("SFT", H.SFT_DIR))
if os.path.exists(H.DPO_DIR):
    STAGES.append(("SFT+DPO", H.DPO_DIR))

for tag, adapter in STAGES:
    processor, model = H.load_model("infer")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).to(model.device)

    print(f"\n===== {tag} × 15 题（逐题输出）=====")
    for q in H.QUESTIONS:
        pred = H.answer(processor, model, H.load_image(q["img"]), q["q"])
        ok = "√" if H.judge(q, pred) else "×"
        print(f"[{q['id']}·{q['type']}] {ok} Q: {q['q']}  A: {pred}   (gold: {q['gold']})")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n提示：把上面各阶段 P01 / P06 / P12 三题的输出抄进报告 4.3 节。")
