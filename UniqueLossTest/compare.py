#!/usr/bin/env python3
import json, os, sys
from io import StringIO

RESULTS_DIR = "./results"

NUM_LETTER_CLASSES = 20
ID2LABEL = {i: chr(ord("A") + i) for i in range(NUM_LETTER_CLASSES)}
ID2LABEL[20] = "single"
# Same as Testing/compare_all.py --eval-classes: A-K (0-10) + single (20)
EVAL_CLASSES = list(range(11)) + [20]
VALID_LABELS = {ID2LABEL[i] for i in EVAL_CLASSES}

def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}_classnms_quality.json")
    with open(path) as f:
        return json.load(f)

def pct(v):
    return f"{float(v)*100:.2f}%" if v is not None else "N/A"

def d(a, b):
    v = float(b) - float(a)
    return f"({'+' if v>=0 else ''}{v:.4f})"

def main():
    a = load("yolo12m_baseline")
    b = load("yolo12m_unique")

    buf = StringIO()
    def p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=buf)

    def filtered_metrics(r):
        pc = r.get("per_class", {})
        tp = sum(pc[c]["tp"] for c in pc if c in VALID_LABELS)
        fp = sum(pc[c]["fp"] for c in pc if c in VALID_LABELS)
        fn = sum(pc[c]["fn"] for c in pc if c in VALID_LABELS)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
        return tp, fp, fn, prec, rec, f1

    atp, afp, afn, aprec, arec, af1 = filtered_metrics(a)
    btp, bfp, bfn, bprec, brec, bf1 = filtered_metrics(b)

    W = 16
    sep = f"  {'-'*68}"

    p(f"\n{'='*70}")
    p(f"  Class-Singleton Suppression Quality — Test Split ({a['images']} images, conf={a['conf']}, iou_nms={a['iou_nms']})")
    p(f"  Classes: A-K + single  ({len(VALID_LABELS)} classes, same as Testing --eval-classes)")
    p(f"{'='*70}")
    p(f"  {'Metric':<18}  {'YOLO12m Baseline':>{W}}  {'YOLO12m+UniqLoss':>{W}}  {'Delta (B-A)':>{W}}")
    p(sep)
    p(f"  {'TP (correct)':<18}  {atp:>{W}}  {btp:>{W}}  {'+' if btp>=atp else ''}{btp-atp:>{W-1}}")
    p(f"  {'FP (wrong box)':<18}  {afp:>{W}}  {bfp:>{W}}  {'+' if bfp>=afp else ''}{bfp-afp:>{W-1}}")
    p(f"  {'FN (missed)':<18}  {afn:>{W}}  {bfn:>{W}}  {'+' if bfn>=afn else ''}{bfn-afn:>{W-1}}")
    p(sep)
    p(f"  {'Precision':<18}  {pct(aprec):>{W}}  {pct(bprec):>{W}}  {d(aprec,bprec):>{W}}")
    p(f"  {'Recall':<18}  {pct(arec):>{W}}  {pct(brec):>{W}}  {d(arec,brec):>{W}}")
    p(f"  {'F1':<18}  {pct(af1):>{W}}  {pct(bf1):>{W}}  {d(af1,bf1):>{W}}")

    # Per-class breakdown
    pc_a = a.get("per_class", {})
    pc_b = b.get("per_class", {})
    if pc_a and pc_b:
        all_cls = sorted((set(pc_a.keys()) | set(pc_b.keys())) & VALID_LABELS)
        p(f"\n  Per-class  (TP / FP / FN)")
        p(f"  {'Class':<8}  {'Base TP':>7} {'Base FP':>7} {'Base FN':>7}  {'Uniq TP':>7} {'Uniq FP':>7} {'Uniq FN':>7}  {'Base F1':>7} {'Uniq F1':>7}")
        p(f"  {'-'*64}")
        for cls in all_cls:
            ra = pc_a.get(cls, {"tp":0,"fp":0,"fn":0})
            rb = pc_b.get(cls, {"tp":0,"fp":0,"fn":0})
            def f1(r):
                pp = r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp']>0 else 0
                rec = r['tp']/(r['tp']+r['fn']) if r['tp']+r['fn']>0 else 0
                return 2*pp*rec/(pp+rec) if pp+rec>0 else 0
            p(f"  {cls:<8}  {ra['tp']:>7} {ra['fp']:>7} {ra['fn']:>7}  "
              f"{rb['tp']:>7} {rb['fp']:>7} {rb['fn']:>7}  "
              f"{f1(ra):>7.3f} {f1(rb):>7.3f}")
    p()

    out_path = os.path.join(RESULTS_DIR, "comparison_classnms.txt")
    with open(out_path, "w") as f:
        f.write(buf.getvalue())
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
