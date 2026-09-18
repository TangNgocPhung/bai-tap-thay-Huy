"""
Demo CLIP zero-shot tren CIFAR-10 va CIFAR-100.

Gop 3 script roi rac (run_cifar_zeroshot.py, run_cifar_zeroshot_timed.py,
run_cifar_transformers.py) thanh mot script duy nhat: chay duoc ca hai bo
du lieu, co prompt ensemble, do thoi gian, ve bieu do top-5, xuat bang
so sanh ra CSV/JSON de dua vao bao cao.

Vi du:
    python demo_clip.py                                  # demo 1 anh, CIFAR-100
    python demo_clip.py --dataset both --eval 10000      # bang so sanh day du
    python demo_clip.py --eval 500 --prompt ensemble     # dung 18 template chuan
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10, CIFAR100

CACHE = os.path.expanduser("~/.cache")

# Mot template duy nhat - cach lam don gian nhat.
SIMPLE_TEMPLATES = ["a photo of a {}."]

# 18 template chinh thuc cua OpenAI cho CIFAR-10/CIFAR-100.
# Nguon: data/prompts.md (muc CIFAR10 va CIFAR100 dung chung bo nay).
OFFICIAL_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

DATASETS = {"cifar10": (CIFAR10, "CIFAR-10"), "cifar100": (CIFAR100, "CIFAR-100")}


def clean(classname):
    """torchvision tra ve 'aquarium_fish', 'maple_tree'... Gach duoi lam hong
    tokenizer nen phai doi thanh dau cach truoc khi ghep vao prompt."""
    return classname.replace("_", " ")


# --------------------------------------------------------------------------- #
# Backend 1: goi truc tiep package `clip` trong repo nay (OpenAI reference impl)
# --------------------------------------------------------------------------- #
class OpenAIBackend:
    def __init__(self, device, arch="ViT-B/32"):
        import clip

        self.clip = clip
        self.device = device
        self.arch = arch
        self.name = f"CLIP {arch}"
        t0 = time.time()
        self.model, self.preprocess = clip.load(arch, device=device)
        self.load_time = time.time() - t0
        self.model.eval()

    @torch.no_grad()
    def _encode_text(self, prompts):
        tokens = self.clip.tokenize(prompts).to(self.device)
        return self.model.encode_text(tokens).float()

    @torch.no_grad()
    def image_features(self, pil_images):
        batch = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        feats = self.model.encode_image(batch).float()
        return feats / feats.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# Backend 2: HuggingFace transformers (openai/clip-vit-base-patch32)
# --------------------------------------------------------------------------- #
class HFBackend:
    def __init__(self, device, arch="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.arch = arch
        self.name = f"transformers {arch.split('/')[-1]}"
        t0 = time.time()
        self.model = CLIPModel.from_pretrained(arch).to(device)
        self.processor = CLIPProcessor.from_pretrained(arch)
        self.load_time = time.time() - t0
        self.model.eval()

    @staticmethod
    def _unwrap(out):
        # transformers >= 5 tra ve BaseModelOutputWithPooling (feature nam o
        # pooler_output); transformers 4.x tra ve thang Tensor.
        return (out if torch.is_tensor(out) else out.pooler_output).float()

    @torch.no_grad()
    def _encode_text(self, prompts):
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._unwrap(self.model.get_text_features(**inputs))

    @torch.no_grad()
    def image_features(self, pil_images):
        inputs = self.processor(images=list(pil_images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self._unwrap(self.model.get_image_features(**inputs))
        return feats / feats.norm(dim=-1, keepdim=True)


BACKENDS = {"openai": OpenAIBackend, "hf": HFBackend}


# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_classifier(backend, classnames, templates, chunk=256):
    """Tao 'trong so classifier' tu chu: moi lop -> trung binh embedding cua
    tat ca template, roi chuan hoa lai. Tra ve ma trix (n_classes, 512)."""
    prompts = [t.format(clean(c)) for c in classnames for t in templates]

    feats = []
    for i in range(0, len(prompts), chunk):
        feats.append(backend._encode_text(prompts[i : i + chunk]))
    feats = torch.cat(feats)

    # (n_classes * n_templates, 512) -> (n_classes, n_templates, 512)
    feats = feats.view(len(classnames), len(templates), -1)
    feats = feats / feats.norm(dim=-1, keepdim=True)  # chuan hoa tung template
    feats = feats.mean(dim=1)  # trung binh cac template
    return feats / feats.norm(dim=-1, keepdim=True)  # chuan hoa lai


def run_single(backend, dataset, tfeat, index, topk):
    """Phan loai zero-shot 1 anh, tra ve top-k nhan + xac suat."""
    image, class_id = dataset[index]

    t0 = time.time()
    ifeat = backend.image_features([image])
    probs = (100.0 * ifeat @ tfeat.T).softmax(dim=-1)[0]
    t_image = time.time() - t0

    values, indices = probs.topk(min(topk, len(dataset.classes)))
    return {
        "index": index,
        "ground_truth": clean(dataset.classes[class_id]),
        "topk": [
            {"label": clean(dataset.classes[i]), "prob": float(v)}
            for v, i in zip(values.tolist(), indices.tolist())
        ],
        "correct": int(indices[0]) == class_id,
        "image_encode_sec": round(t_image, 3),
    }, image


@torch.no_grad()
def run_eval(backend, dataset, tfeat, n, batch_size):
    """Do accuracy zero-shot tren n anh dau tien cua tap test."""
    n = min(n, len(dataset))
    k5 = min(5, len(dataset.classes))
    top1 = top5 = 0
    t0 = time.time()

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        chunk = [dataset[i] for i in range(start, stop)]
        images = [c[0] for c in chunk]
        labels = torch.tensor([c[1] for c in chunk], device=tfeat.device)

        logits = 100.0 * backend.image_features(images) @ tfeat.T
        pred = logits.topk(k5, dim=-1).indices
        top1 += (pred[:, 0] == labels).sum().item()
        top5 += (pred == labels[:, None]).any(dim=-1).sum().item()

        done = stop
        rate = done / (time.time() - t0)
        eta = (n - done) / rate
        print(
            f"  {done}/{n} anh | {rate:.1f} anh/s | con ~{eta / 60:.1f} phut   ",
            end="\r",
            flush=True,
        )

    elapsed = time.time() - t0
    print(" " * 60, end="\r")
    return {
        "n_images": n,
        "top1_acc": round(100 * top1 / n, 2),
        "top5_acc": round(100 * top5 / n, 2),
        "total_sec": round(elapsed, 2),
        "sec_per_image": round(elapsed / n, 4),
    }


def save_figure(image, result, dataset_label, path):
    """Ve anh goc + bieu do cot ngang top-k, luu PNG de chen vao bao cao."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [d["label"] for d in result["topk"]][::-1]
    probs = [100 * d["prob"] for d in result["topk"]][::-1]
    colors = ["#4C72B0"] * len(labels)
    colors[-1] = "#2E8B57" if result["correct"] else "#C44E52"

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 2]}
    )
    ax_img.imshow(image)
    ax_img.set_title(
        f"{dataset_label} #{result['index']}\ntruth: {result['ground_truth']}",
        fontsize=10,
    )
    ax_img.axis("off")

    bars = ax_bar.barh(labels, probs, color=colors)
    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel("probability (%)")
    ax_bar.set_title("CLIP zero-shot top predictions", fontsize=10)
    for bar, p in zip(bars, probs):
        ax_bar.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.2f}%",
            va="center",
            fontsize=9,
        )
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_table(rows):
    """In bang so sanh dang text, canh cot theo do rong thuc te."""
    head = [
        "Dataset",
        "Classes",
        "Test Images",
        "Model",
        "Method",
        "Top-1 (%)",
        "Top-5 (%)",
        "Time (s)",
    ]
    body = [[str(c) for c in r] for r in rows]
    width = [max(len(h), *(len(r[i]) for r in body)) for i, h in enumerate(head)]

    line = "  ".join(h.ljust(w) for h, w in zip(head, width))
    print(line)
    print("-" * len(line))
    for r in body:
        print("  ".join(c.ljust(w) for c, w in zip(r, width)))


def main():
    ap = argparse.ArgumentParser(description="Demo CLIP zero-shot tren CIFAR")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "both"],
                    default="cifar100")
    ap.add_argument("--index", type=int, default=3637, help="chi so anh trong tap test")
    ap.add_argument("--backend", choices=["openai", "hf", "both"], default="openai")
    ap.add_argument("--arch", default="ViT-B/32", help="chi ap dung cho backend openai")
    ap.add_argument("--prompt", choices=["simple", "ensemble"], default="simple",
                    help="simple = 1 template; ensemble = 18 template chuan")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--eval", type=int, default=0,
                    help="so anh de do accuracy zero-shot (0 = bo qua)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-figure", action="store_true", help="bo qua ve bieu do")
    ap.add_argument("--out", default="demo_out", help="thu muc xuat ket qua")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    templates = OFFICIAL_TEMPLATES if args.prompt == "ensemble" else SIMPLE_TEMPLATES
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print(f"Device   : {device}")
    print(f"Torch    : {torch.__version__}")
    print(f"Prompt   : {args.prompt} ({len(templates)} template)")

    ds_names = ["cifar10", "cifar100"] if args.dataset == "both" else [args.dataset]
    be_names = ["openai", "hf"] if args.backend == "both" else [args.backend]

    report = {"device": device, "torch": torch.__version__,
              "prompt_mode": args.prompt, "runs": []}
    table = []

    for be_name in be_names:
        kwargs = {"arch": args.arch} if be_name == "openai" else {}
        backend = BACKENDS[be_name](device, **kwargs)
        print(f"\nModel    : {backend.name} (load {backend.load_time:.1f}s)")

        for ds_name in ds_names:
            ds_cls, ds_label = DATASETS[ds_name]
            print("\n" + "=" * 64)
            print(f"{ds_label}  |  backend: {be_name}")
            print("=" * 64)

            dataset = ds_cls(root=CACHE, download=True, train=False)
            n_class = len(dataset.classes)
            print(f"Tap test : {len(dataset)} anh, {n_class} lop")

            t0 = time.time()
            tfeat = build_classifier(backend, dataset.classes, templates)
            t_text = time.time() - t0
            print(f"Encode text: {n_class} lop x {len(templates)} template "
                  f"= {n_class * len(templates)} prompt trong {t_text:.1f}s")

            run = {"dataset": ds_label, "backend": be_name, "model": backend.name,
                   "n_classes": n_class, "load_time_sec": round(backend.load_time, 2),
                   "text_encode_sec": round(t_text, 2)}

            idx = min(args.index, len(dataset) - 1)
            result, image = run_single(backend, dataset, tfeat, idx, args.topk)
            print(f"\nAnh #{idx} - nhan that: {result['ground_truth']}")
            for d in result["topk"]:
                mark = "<-- dung" if d["label"] == result["ground_truth"] else ""
                print(f"  {d['label']:>16s}: {100 * d['prob']:6.2f}%  {mark}")
            run["single"] = result

            if not args.no_figure:
                fig = out / f"topk_{ds_name}_{be_name}_{idx}.png"
                save_figure(image, result, ds_label, fig)
                print(f"Bieu do -> {fig}")

            if args.eval > 0:
                print(f"\nDo accuracy tren {min(args.eval, len(dataset))} anh...")
                m = run_eval(backend, dataset, tfeat, args.eval, args.batch_size)
                print(f"  Top-1: {m['top1_acc']}%   Top-5: {m['top5_acc']}%")
                print(f"  {m['total_sec']}s ({m['sec_per_image']}s/anh)")
                run["eval"] = m
                table.append([ds_label, n_class, m["n_images"], backend.name,
                              "Zero-shot", m["top1_acc"], m["top5_acc"],
                              m["total_sec"]])

            report["runs"].append(run)

    if table:
        print("\n" + "=" * 64)
        print("BANG SO SANH")
        print("=" * 64)
        print_table(table)

        csv_path = out / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Dataset", "Classes", "Test Images", "Model", "Method",
                        "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Time (s)"])
            w.writerows(table)
        print(f"\nCSV -> {csv_path}")

    json_path = out / "results.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"JSON -> {json_path}")


if __name__ == "__main__":
    main()
