# -*- coding: utf-8 -*-
"""
SmolVLM-256M QLoRA 微调 + DPO 偏好对齐 —— 实验作业五（修订版）

对应教材「参数高效微调」与「偏好对齐 / DPO」相关章节。
单文件、开箱即跑：内置 PIL 合成「几何图形」图像 15 张、问答 15 题、
SFT 样本 15 条、偏好对 12 组 + 自拟 5 组（任务三）。

常用命令：
    python hw5_dpo.py --mode make_data   # 生成 15 张合成图像（首次运行自动执行）
    python hw5_dpo.py --mode infer       # 基线推理 15 题，记录基线正确率
    python hw5_dpo.py --mode sft         # QLoRA SFT（产出 ./sft_adapter）
    python hw5_dpo.py --mode dpo         # 在 SFT 之上做 DPO（产出 ./dpo_adapter）
    python hw5_dpo.py --mode eval        # 三阶段（基线/SFT/SFT+DPO）× 15 题对比
    python hw5_dpo.py --dpo_check 1.2,0.8,0.9,0.6,0.1   # DPO loss 手算复核

本文件基于指导书第七节参考代码，修正了 5 处问题（详见文件末尾「修订说明」）。
"""

import argparse
import math
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMG_DIR = "hw5_imgs"
SFT_DIR, DPO_DIR = "sft_adapter", "dpo_adapter"

# 模型是否走了 4-bit 量化路径（load_model 里设置，供日志与报告注明用）
MODEL_INFO = {"quantized": False}


# ============================================================
# 1. 合成图像库「几何图形」：15 张图，三类问题的答案均已埋入
#    每张图 = 白底 256x256，画若干「颜色-形状」对（位置固定可复现）
# ============================================================
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}

# img_id -> [(颜色, 形状, 个数), ...]（形状：圆=ellipse，方=rectangle）
IMG_SPEC = {
    "img01": [("红", "圆", 2), ("蓝", "方", 1)],
    "img02": [("绿", "圆", 3)],
    "img03": [("蓝", "圆", 1), ("黄", "方", 2)],
    "img04": [("红", "方", 3)],
    "img05": [("绿", "方", 2), ("红", "圆", 1)],
    "img06": [("黄", "圆", 4)],
    "img07": [("蓝", "方", 2), ("绿", "圆", 1)],
    "img08": [("红", "圆", 4)],
    "img09": [("绿", "方", 1), ("黄", "圆", 2)],
    "img10": [("蓝", "圆", 3)],
    "img11": [("黄", "方", 1)],                     # 只有一个黄方 → P11 答「有」
    "img12": [("红", "方", 1), ("蓝", "方", 1)],    # 没有圆形 → P12「有没有红圆」答「没有」
    "img13": [("绿", "圆", 2), ("黄", "圆", 1)],
    "img14": [("蓝", "方", 3), ("红", "方", 1)],
    "img15": [("黄", "圆", 2), ("绿", "圆", 2)],    # 没有蓝色 → P15 答「没有」
}


def make_images():
    os.makedirs(IMG_DIR, exist_ok=True)
    rnd = random.Random(42)
    for img_id, spec in IMG_SPEC.items():
        im = Image.new("RGB", (256, 256), "white")
        d = ImageDraw.Draw(im)
        for color, shape, n in spec:
            for _ in range(n):
                x, y = rnd.randint(10, 200), rnd.randint(10, 200)
                s = rnd.randint(30, 50)
                box = [x, y, x + s, y + s]
                if shape == "圆":
                    d.ellipse(box, outline=COLORS[color], width=4)
                else:
                    d.rectangle(box, outline=COLORS[color], width=4)
        im.save(f"{IMG_DIR}/{img_id}.png")
    print(f"[data] 已生成 {len(IMG_SPEC)} 张图像到 {IMG_DIR}/")


def load_image(img_id):
    p = f"{IMG_DIR}/{img_id}.png"
    if not os.path.exists(p):
        make_images()
    return Image.open(p).convert("RGB")


# ============================================================
# 2. 问题集：3 类 × 5 题（gold = 标准答案；与参考答案逐字匹配计分）
#    count=数数（局部事实） attr=属性识别 exist=存在性/组合判断
# ============================================================
QUESTIONS = [
    {"id": "P01", "type": "count", "img": "img01", "q": "图中有几个红色圆形？", "gold": "2"},
    {"id": "P02", "type": "count", "img": "img02", "q": "图中有几个绿色圆形？", "gold": "3"},
    {"id": "P03", "type": "count", "img": "img03", "q": "图中有几个黄色方形？", "gold": "2"},
    {"id": "P04", "type": "count", "img": "img06", "q": "图中有几个黄色圆形？", "gold": "4"},
    {"id": "P05", "type": "count", "img": "img10", "q": "图中有几个蓝色圆形？", "gold": "3"},
    {"id": "P06", "type": "attr", "img": "img01", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P07", "type": "attr", "img": "img05", "q": "图中红色的图形是什么形状？", "gold": "圆形"},
    {"id": "P08", "type": "attr", "img": "img09", "q": "图中绿色的图形是什么形状？", "gold": "方形"},
    {"id": "P09", "type": "attr", "img": "img13", "q": "图中黄色的图形是什么形状？", "gold": "圆形"},
    {"id": "P10", "type": "attr", "img": "img14", "q": "图中蓝色的图形是什么形状？", "gold": "方形"},
    {"id": "P11", "type": "exist", "img": "img11", "q": "图里有没有黄色方形？", "gold": "有"},
    {"id": "P12", "type": "exist", "img": "img12", "q": "图里有没有红色圆形？", "gold": "没有"},
    {"id": "P13", "type": "exist", "img": "img07", "q": "图里有没有绿色圆形？", "gold": "有"},
    {"id": "P14", "type": "exist", "img": "img08", "q": "图里有没有绿色方形？", "gold": "没有"},
    {"id": "P15", "type": "exist", "img": "img15", "q": "图里有没有蓝色图形？", "gold": "没有"},
]

# SFT 数据：图像 + 问题 → 简短标准答案（教模型「答题格式 + 事实」）
SFT_DATA = [{"img": q["img"], "prompt": q["q"],
             "answer": f"{q['gold']}。" if q["type"] != "count" else f"{q['gold']}个。"}
            for q in QUESTIONS]

# 偏好对（12 组）：chosen = 正确回答；rejected = 典型幻觉/错误
# rejected 的三类错误：数错（off-by-one / 混淆）、形状幻觉、对不存在的物体硬答「有」
PRE_PAIRS = [
    {"img": "img01", "prompt": "图中有几个红色圆形？",
     "chosen": "2 个。", "rejected": "3 个。图中有三个红色圆形整齐排列。"},
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "2 个。"},
    {"img": "img03", "prompt": "图中有几个黄色方形？",
     "chosen": "2 个。", "rejected": "没有看到黄色图形。"},
    {"img": "img06", "prompt": "图中有几个黄色圆形？",
     "chosen": "4 个。", "rejected": "3 个。"},
    {"img": "img10", "prompt": "图中有几个蓝色圆形？",
     "chosen": "3 个。", "rejected": "图中有三个蓝色方形。"},
    {"img": "img01", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。", "rejected": "圆形。"},
    {"img": "img05", "prompt": "图中红色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形，而且有两个红色方形。"},
    {"img": "img09", "prompt": "图中绿色的图形是什么形状？",
     "chosen": "方形。", "rejected": "图中没有绿色图形。"},
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有。", "rejected": "没有。"},
    {"img": "img12", "prompt": "图里有没有红色圆形？",
     "chosen": "没有。", "rejected": "有，图左上角有一个红色圆形。"},
    {"img": "img08", "prompt": "图里有没有绿色方形？",
     "chosen": "没有。", "rejected": "有，一共两个绿色方形。"},
    {"img": "img15", "prompt": "图里有没有蓝色图形？",
     "chosen": "没有。图中只有黄色和绿色的圆形。", "rejected": "有，两个蓝色圆形。"},
]

# ------------------------------------------------------------------
# 任务三：自拟 5 组偏好对（追加进 PRE_PAIRS，合计 12 + 5 = 17 组）
# chosen 全部来自标准答案；rejected 各自对应一种典型失败模式，
# 见每组上方的注释。想对比「12 组 vs 17 组」的效果时，
# 把下面这行注释掉即可回到原始 12 组。
# ------------------------------------------------------------------
MY_PAIRS = [
    # ① 形状幻觉：img13 里黄色图形是「圆形」，rejected 说成「方形」
    {"img": "img13", "prompt": "图中黄色的图形是什么形状？",
     "chosen": "圆形。", "rejected": "方形。"},

    # ② 冗长废话型：先绕一大圈，最后仍给出错误形状
    #    （img14 里蓝色图形是「方形」）
    {"img": "img14", "prompt": "图中蓝色的图形是什么形状？",
     "chosen": "方形。",
     "rejected": "这张图整体构图比较简洁，画面里有若干个不同颜色的几何图形，"
                 "我先把每个图形都扫了一遍。就你问的这个蓝色图形而言，"
                 "它看起来更接近圆形一些。"},

    # ③ 反向幻觉：img07 里明明有一个绿色圆形，rejected 硬答「没有」
    #    （PRE_PAIRS 只覆盖了「硬答有」，这里补上相反方向，
    #      正好针对基线在 exist 题上的整体 No 偏置）
    {"img": "img07", "prompt": "图里有没有绿色圆形？",
     "chosen": "有。", "rejected": "没有，图里只有蓝色的图形。"},

    # ④ 属性混淆：img02 里是三个「绿色」圆形，rejected 数量对、颜色错
    #    （比单纯数错更细：模型得同时绑定颜色和形状）
    {"img": "img02", "prompt": "图中有几个绿色圆形？",
     "chosen": "3 个。", "rejected": "图中有三个蓝色圆形。"},

    # ⑤ 结论对、细节编造：img11 里只有一个黄色方形
    #    （对应基线最典型的一类失败——「答案对但理由是编的」）
    {"img": "img11", "prompt": "图里有没有黄色方形？",
     "chosen": "有，图里有一个黄色方形。",
     "rejected": "有，图里有两个黄色方形。"},
]

PRE_PAIRS = PRE_PAIRS + MY_PAIRS


# ============================================================
# 3. 模型加载：QLoRA（教材：NF4 量化 + LoRA 低秩适配器）
#    256M 小模型按理可全参微调，但本实验统一走 QLoRA 流程以体会方法
# ============================================================
def _bnb_ok():
    """4-bit NF4 需要 bitsandbytes + NVIDIA CUDA，两者缺一不可。"""
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _dtype_kwargs():
    """transformers 4.x 用 torch_dtype，5.x 起改名为 dtype（顺带消掉 FutureWarning 刷屏）。"""
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": torch.float32} if major >= 5 else {"torch_dtype": torch.float32}


def _model_cls():
    """图像-文本模型的自动类：4.x 叫 AutoModelForVision2Seq，5.x 改名为 AutoModelForImageTextToText。"""
    import transformers
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("当前 transformers 中找不到图像-文本模型自动类，请检查版本")


def load_model(mode="infer"):
    """mode: infer=原精度推理 / train=4bit 量化 + LoRA"""
    from transformers import AutoProcessor

    auto_cls = _model_cls()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dtype_kw = _dtype_kwargs()

    quantized = (mode == "train") and _bnb_ok()
    MODEL_INFO["quantized"] = quantized

    if quantized:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
        )
        # 注意：bitsandbytes 4-bit 模型不支持 .to()，必须用 device_map 直接放到 GPU
        model = auto_cls.from_pretrained(
            MODEL_ID, quantization_config=qcfg, device_map={"": "cuda"}, **dtype_kw)
        print("[model] 4-bit NF4 量化基座 + LoRA（QLoRA 路径）")
    else:
        if mode == "train":
            print("[warn] 未检测到 bitsandbytes/CUDA，已回退 fp32 基座 + LoRA（报告中请注明）")
        model = auto_cls.from_pretrained(MODEL_ID, **dtype_kw)
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "train":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if quantized:
            model = prepare_model_for_kbit_training(model)
        else:
            for p in model.parameters():          # 冻结基座：只训适配器
                p.requires_grad = False
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()    # ← 报告 2.1 节要抄的可训练参数占比

    return processor, model


def build_prompt(processor, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    try:
        return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except TypeError:
        return processor.apply_chat_template(msgs, add_generation_prompt=True)


@torch.no_grad()
def answer(processor, model, img, q, max_new=24):
    inputs = processor(text=[build_prompt(processor, q)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def judge(q, pred):
    """宽松判分：标准答案出现在预测里即算对（「没有」须整词出现）"""
    p, g = pred.replace(" ", ""), q["gold"]
    if g == "没有":
        return "没有" in p
    if g == "有":
        # 修正：字符串「没有」包含「有」，原实现会把答错「没有」判成正确
        return ("有" in p) and ("没有" not in p)
    return g in p


# ============================================================
# 4. SFT（教材：监督微调）：只对答案 token 计损失，全流程手写便于理解
# ============================================================
def sft_step(model, processor, opt, epochs=2, steps=60, log_every=10):
    dev = next(model.parameters()).device
    model.train()
    done = False
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(SFT_DATA)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            step_no = ep * len(order) + i + 1
            if step_no > steps:
                print(f"[sft] 已达 {steps} 步，提前结束")
                done = True
                break

            s = SFT_DATA[idx]
            img = load_image(s["img"])
            full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                             images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                prompt_len = processor(text=[build_prompt(processor, s["prompt"])],
                                       images=[img],
                                       return_tensors="pt")["input_ids"].shape[1]

            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100          # 只学答案
            full["labels"] = labels
            loss = model(**full).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            if step_no % log_every == 0 or step_no >= steps:
                print(f"[sft] ep{ep+1} step{step_no}/{min(steps, epochs * len(order))} "
                      f"loss={loss.item():.4f}")
        if done:
            break
    model.save_pretrained(SFT_DIR)
    print(f"[sft] 适配器已保存到 ./{SFT_DIR}")


# ============================================================
# 5. DPO（教材：直接偏好优化）
#    loss = -log σ( β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] )
#    参考模型 = 关闭 LoRA 适配器后的基座（disable_adapter 上下文）
# ============================================================
def seq_logprob(model, processor, img, prompt, resp):
    """返回「答案 token 的序列总对数概率」= Σ log πθ(token | 上文)"""
    dev = next(model.parameters()).device
    full = processor(text=[build_prompt(processor, prompt) + resp],
                     images=[img], return_tensors="pt").to(dev)
    plen = processor(text=[build_prompt(processor, prompt)],
                     images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    logits = model(**full).logits[:, :-1, :]
    tgt = labels[:, 1:]
    lp = torch.log_softmax(logits.float(), dim=-1)
    mask = tgt.ne(-100)
    return lp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum()


def dpo_step(model, processor, opt, beta=0.1, epochs=1, steps=40, log_every=5):
    dev = next(model.parameters()).device
    model.train()
    done = False
    for ep in range(epochs):
        rnd = random.Random(ep)
        order = list(range(len(PRE_PAIRS)))
        rnd.shuffle(order)
        for i, idx in enumerate(order):
            step_no = ep * len(order) + i + 1
            if step_no > steps:
                print(f"[dpo] 已达 {steps} 步，提前结束")
                done = True
                break

            p = PRE_PAIRS[idx]
            img = load_image(p["img"])
            pol_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
            pol_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])
            with torch.no_grad(), model.disable_adapter():     # 冻结的参考模型 π_ref
                ref_w = seq_logprob(model, processor, img, p["prompt"], p["chosen"])
                ref_l = seq_logprob(model, processor, img, p["prompt"], p["rejected"])

            margin = beta * ((pol_w - ref_w) - (pol_l - ref_l))
            loss = -F.logsigmoid(margin)
            loss.backward()
            opt.step()
            opt.zero_grad()
            # 每步都打印：报告需要完整的 loss / margin 曲线
            print(f"[dpo] step{step_no} loss={loss.item():.4f} margin={margin.item():.3f}")
        if done:
            break
    model.save_pretrained(DPO_DIR)
    print(f"[dpo] 适配器已保存到 ./{DPO_DIR}")


def dpo_check(arg):
    """手算复核：依次给 pol_w, ref_w, pol_l, ref_l, beta"""
    pw, rw, pl, rl, beta = [float(x) for x in arg.split(",")]
    margin = beta * ((pw - rw) - (pl - rl))
    loss = math.log(1 + math.exp(-margin)) if margin >= 0 else -margin + math.log(1 + math.exp(margin))
    print(f"margin = β·[(πθ(yw)-πref(yw)) - (πθ(yl)-πref(yl))] = {margin:.4f}")
    print(f"DPO loss = -log σ(margin) = {loss:.4f}"
          f"（margin 越大 loss 越小；margin<0 说明模型更偏好 rejected，需要继续训练）")


# ============================================================
# 6. 三阶段对比评测：基线 / SFT / SFT+DPO × 15 题
# ============================================================
def evaluate(processor, model, tag):
    print(f"\n===== {tag} × 15 题 =====")
    ok_by_type = {}
    for tp in ("count", "attr", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = sum(judge(q, answer(processor, model, load_image(q["img"]), q["q"])) for q in qs)
        ok_by_type[tp] = (ok, len(qs))
        print(f"{tag:<10}{tp:<8}正确 {ok}/{len(qs)}")
    total = sum(o for o, _ in ok_by_type.values())
    print(f"{tag:<10}总计         正确 {total}/15\n")
    return total


# ============================================================
# 7. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业五：SmolVLM-256M QLoRA + DPO")
    ap.add_argument("--mode", choices=["make_data", "infer", "sft", "dpo", "eval"], default="infer")
    ap.add_argument("--steps", type=int, default=None, help="覆盖默认训练步数（CPU 演示用 5~10）")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数 β")
    ap.add_argument("--dpo_check", default=None, help="如 1.2,0.8,0.9,0.6,0.1")
    args = ap.parse_args()

    if args.dpo_check:
        dpo_check(args.dpo_check)
        return

    if args.mode == "make_data":
        make_images()
        return

    if args.mode == "infer":
        processor, model = load_model("infer")
        for q in QUESTIONS:
            pred = answer(processor, model, load_image(q["img"]), q["q"])
            mark = "√" if judge(q, pred) else "×"
            print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q']} A: {pred}（gold: {q['gold']}）")
        return

    if args.mode == "sft":
        processor, model = load_model("train")
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        steps = args.steps or 60
        sft_step(model, processor, opt, steps=steps, log_every=max(1, steps // 6))
        return

    if args.mode == "dpo":
        if not os.path.exists(SFT_DIR):
            raise SystemExit("先运行 --mode sft 生成 SFT 适配器")
        processor, model = load_model("train")
        from peft import PeftModel
        # 直接在已加载的基座上挂 SFT 适配器继续训练；
        # 不要对 4-bit 基座调用 .float() / .to()，bitsandbytes 不支持
        base = model.get_base_model()
        model = PeftModel.from_pretrained(base, SFT_DIR, is_trainable=True)
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
        dpo_step(model, processor, opt, beta=args.beta, steps=args.steps or 40)
        return

    if args.mode == "eval":
        processor, model = load_model("infer")
        evaluate(processor, model, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, SFT_DIR)
            evaluate(processor, model, "SFT")
            if os.path.exists(DPO_DIR):
                # 同一个 PeftModel 上再挂一个适配器，避免嵌套 PeftModel
                model.load_adapter(DPO_DIR, adapter_name="dpo")
                model.set_adapter("dpo")
                evaluate(processor, model, "SFT+DPO")
        print("提示：把三阶段结果抄进报告表 3；另抽 2 题做人工忠实度评分（答案是否仅陈述图中事实）。")


if __name__ == "__main__":
    main()


# ============================================================
# 修订说明（相对指导书第七节参考代码，共 5 处）
#
# [1] judge()：gold == "有" 时原实现为 `return g in p`，而字符串「没有」包含
#     「有」，导致模型答错「没有」也被判为正确 —— 会直接污染三阶段正确率。
#     现改为：出现「有」且不出现「没有」才算对。
#
# [2] sft_step() / dpo_step()：原实现在步数上限处直接 `return`，
#     跳过了函数末尾的 model.save_pretrained(...)。因此 CPU 上跑
#     `--steps 10`（SFT）、`--steps 2`（DPO）时适配器根本不会落盘，
#     后续 --mode eval 永远只显示「基线」一个阶段。
#     现改为 break 出循环、统一在循环后保存。
#
# [3] load_model()：4-bit NF4 需要 bitsandbytes + CUDA。原实现无条件构造
#     BitsAndBytesConfig，无 GPU 时在加载模型处直接抛异常。
#     现增加 _bnb_ok() 检测，不满足时回退 fp32 基座 + LoRA 并打印告警。
#
# [4] load_model()：4-bit 模型不支持 .to()/.float()（bitsandbytes 限制），
#     原实现末尾统一 model.to(...) 在量化路径下会报错。
#     现量化路径改用 device_map={"": "cuda"} 装载，路径分开处理。
#
# [5] --mode dpo：原实现 model.get_base_model().float().to("cpu") 对量化基座
#     会抛出「.to is not supported for 4bit」；--mode eval 第三阶段用
#     PeftModel.from_pretrained 在已有 PeftModel 上再包一层，同样会失败。
#     现分别改为「直接挂载」与 load_adapter()/set_adapter()。
#
# 另有两处不影响正确性的小调整：
#   - _dtype_kwargs()：按 transformers 主版本选择 torch_dtype / dtype，
#     消除 4.57 上的 FutureWarning 刷屏。
#   - DPO 改为每步打印 loss / margin（原本 log_every=5，CPU 只跑 2 步时
#     一行都不打印，报告拿不到曲线数据）。
#
# [6] load_model()：transformers 5.x 删除了 AutoModelForVision2Seq，
#     改名为 AutoModelForImageTextToText（4.x 两个名字都在）。
#     现用 _model_cls() 自动挑选，4.x / 5.x 都能跑。
# ============================================================
# ============================================================
