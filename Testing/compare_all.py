#!/usr/bin/env python3
"""
Full model comparison — YOLO (all variants) + DAB-DETR, detector-only.
Use this to isolate pure detector performance.

Outputs (in --out-dir):
  <model_name>/
      per_class.csv        per-class P/R/F1/AP + TP/FP/FN
      confusion_matrix.csv NxN confusion matrix
      results.json         full numeric record
  comparison_summary.csv   one row per model (incl. timing, params, GFLOPs, CI)

Usage:
    python compare_all.py
    python compare_all.py --conf 0.3 --iou 0.5
    python compare_all.py --max-samples 50
    python compare_all.py --bootstrap 1000
"""

import os, csv, gc, json, glob, argparse, time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ── constants ──────────────────────────────────────────────────────────────────
NUM_LETTER_CLASSES = 20
NUM_PANEL_CLASSES  = 21
BG_IDX             = NUM_PANEL_CLASSES
CONF_SIZE          = NUM_PANEL_CLASSES + 1

ID2LABEL = {i: chr(ord("A") + i) for i in range(NUM_LETTER_CLASSES)}
ID2LABEL[20] = "single"

IMAGE_EXTS    = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp")
YOLO_REGISTRY = "../ModelBench/runs/detect/runs_mydata"
DABDETR_ROOT  = "../ModelBench/checkpoints_mydata/dabdetr"

# Classes with enough samples to be meaningful (A–I = 0–8, single = 20)
EVAL_CLASSES  = list(range(11)) + [20]


# ── IoU ────────────────────────────────────────────────────────────────────────
def box_iou(a, b):
    ax1,ay1,ax2,ay2 = a[:,0],a[:,1],a[:,2],a[:,3]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    ix1 = torch.max(ax1.unsqueeze(1), bx1.unsqueeze(0))
    iy1 = torch.max(ay1.unsqueeze(1), by1.unsqueeze(0))
    ix2 = torch.min(ax2.unsqueeze(1), bx2.unsqueeze(0))
    iy2 = torch.min(ay2.unsqueeze(1), by2.unsqueeze(0))
    inter  = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    return inter / (area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter).clamp(1e-6)


# ── GT ────────────────────────────────────────────────────────────────────────
def load_gt(label_path, W, H):
    boxes, classes = [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls = int(parts[0])
            if cls >= NUM_PANEL_CLASSES: continue
            xc, yc, bw, bh = map(float, parts[1:5])
            boxes.append([(xc-bw/2)*W,(yc-bh/2)*H,(xc+bw/2)*W,(yc+bh/2)*H])
            classes.append(cls)
    return boxes, classes


# ── model discovery ────────────────────────────────────────────────────────────
def find_yolo_models():
    return [{"name": Path(pt).parent.parent.name, "type": "yolo", "ckpt": pt}
            for pt in sorted(glob.glob(
                os.path.join(YOLO_REGISTRY, "*", "weights", "best.pt")))]

def resolve_dabdetr(root):
    best_txt = os.path.join(root, "best_path.txt")
    if os.path.isfile(best_txt):
        rel = open(best_txt).read().strip()
        abs_root = os.path.abspath(root)
        for n_up in (2, 1):
            base = abs_root
            for _ in range(n_up):
                base = os.path.dirname(base)
            ckpt = os.path.normpath(os.path.join(base, rel))
            if os.path.isdir(ckpt):
                return ckpt
    return None


# ── model profiling (params + GFLOPs) ─────────────────────────────────────────
def profile_model(model_info, device, img_size=(640, 640)):
    """Returns (params_M, gflops). GFLOPs via thop (CPU pass to avoid device issues)."""
    mtype = model_info["type"]
    params_M = -1.0
    gflops   = -1.0

    try:
        import thop
        if mtype == "yolo":
            from ultralytics import YOLO
            m = YOLO(model_info["ckpt"])
            m.model.eval()
            params_M = sum(p.numel() for p in m.model.parameters()) / 1e6
            dummy = torch.zeros(1, 3, *img_size)          # CPU — matches uninitialised model
            flops, _ = thop.profile(m.model, inputs=(dummy,), verbose=False)
            gflops = flops / 1e9
            del m
        else:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            m = AutoModelForObjectDetection.from_pretrained(model_info["ckpt"])
            m.eval()
            params_M = sum(p.numel() for p in m.parameters()) / 1e6
            proc = AutoImageProcessor.from_pretrained(model_info["ckpt"])
            dummy_pil = Image.new("RGB", img_size)
            enc = proc(images=dummy_pil, return_tensors="pt")
            pv  = enc["pixel_values"]                      # CPU
            # thop needs positional args — wrap model so pixel_values is positional
            class _W(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, pv): return self.m(pixel_values=pv)
            flops, _ = thop.profile(_W(m), inputs=(pv,), verbose=False)
            gflops = flops / 1e9
            del m
        gc.collect()
    except ImportError:
        print("  [profile] thop not found — pip install thop")
    except Exception as e:
        print(f"  [profile warn] {model_info['name']}: {e}")

    return round(params_M, 2), round(gflops, 2)


# ── bootstrap 95% CI ──────────────────────────────────────────────────────────
def bootstrap_ci(values, n_boot=1000, ci=95):
    """Returns (mean, lower, upper) bootstrap CI for a list of per-sample values."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    means = np.array([np.mean(np.random.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(np.mean(arr)), float(lo), float(hi)


# ── evaluate one model ────────────────────────────────────────────────────────
def eval_model(model_info, test_samples, args, has_tm, MAP):
    tp_map   = defaultdict(int)
    fp_map   = defaultdict(int)
    fn_map   = defaultdict(int)
    conf_mat = np.zeros((CONF_SIZE, CONF_SIZE), dtype=np.int32)

    metric   = MAP(iou_thresholds=[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],
                   class_metrics=True) if has_tm else None
    metric50 = MAP(iou_thresholds=[0.50], class_metrics=True) if has_tm else None
    metric75 = MAP(iou_thresholds=[0.75], class_metrics=True) if has_tm else None

    # per-image accumulators for CI and timing
    per_img_p, per_img_r, per_img_f1 = [], [], []
    infer_times_ms = []

    use_cuda = args.device.type == "cuda"

    mtype = model_info["type"]
    if mtype == "yolo":
        from ultralytics import YOLO
        detector = YOLO(model_info["ckpt"]); detector.to(args.device)
    else:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        from torchvision.ops import batched_nms
        processor = AutoImageProcessor.from_pretrained(model_info["ckpt"])
        detector  = AutoModelForObjectDetection.from_pretrained(
            model_info["ckpt"]).to(args.device); detector.eval()

    # warmup (3 frames) so first-image latency doesn't skew timing
    warmup_samples = test_samples[:3]
    for img_path, pil, _, _ in warmup_samples:
        try:
            if mtype == "yolo":
                detector.predict(str(img_path), conf=args.conf, iou=args.iou, verbose=False)
            else:
                enc = processor(images=pil, return_tensors="pt")
                pv  = enc["pixel_values"].to(args.device)
                kw  = {"pixel_values": pv}
                if "pixel_mask" in enc: kw["pixel_mask"] = enc["pixel_mask"].to(args.device)
                with torch.no_grad(): detector(**kw)
        except Exception:
            pass

    n_evaluated = 0
    for img_path, pil, gt_boxes, gt_classes in tqdm(
            test_samples, desc=model_info["name"], leave=False):
        W, H = pil.size
        try:
            if use_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()

            if mtype == "yolo":
                results = detector.predict(str(img_path),
                    conf=args.conf, iou=args.iou, verbose=False)
                det_boxes, det_scores, det_labels = [], [], []
                for r in results:
                    if r.boxes is None: continue
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        if cls_id >= NUM_PANEL_CLASSES: continue
                        det_boxes.append(box.xyxy[0].tolist())
                        det_scores.append(float(box.conf.item()))
                        det_labels.append(cls_id)
            else:
                enc = processor(images=pil, return_tensors="pt")
                pv  = enc["pixel_values"].to(args.device)
                kw  = {"pixel_values": pv}
                if "pixel_mask" in enc:
                    kw["pixel_mask"] = enc["pixel_mask"].to(args.device)
                with torch.no_grad():
                    outs = detector(**kw)
                res = processor.post_process_object_detection(
                    outs, threshold=args.conf, target_sizes=[(H,W)])[0]
                pb = res["boxes"].cpu(); ps = res["scores"].cpu(); pl = res["labels"].cpu()
                if pb.numel() > 0:
                    keep = batched_nms(pb, ps, pl, args.iou)
                    pb=pb[keep]; ps=ps[keep]; pl=pl[keep]
                mask = pl < NUM_PANEL_CLASSES
                det_boxes=pb[mask].tolist(); det_scores=ps[mask].tolist(); det_labels=pl[mask].tolist()

            if use_cuda: torch.cuda.synchronize()
            infer_times_ms.append((time.perf_counter() - t0) * 1000)

        except Exception as e:
            print(f"    [warn] {os.path.basename(str(img_path))}: {e}")
            continue

        gt_boxes_t  = torch.tensor(gt_boxes,   dtype=torch.float32)
        gt_labels_t = torch.tensor(gt_classes, dtype=torch.long)
        if det_boxes:
            pred_boxes  = torch.tensor(det_boxes,  dtype=torch.float32)
            pred_scores = torch.tensor(det_scores, dtype=torch.float32)
            pred_labels = torch.tensor(det_labels, dtype=torch.long)
        else:
            pred_boxes=torch.zeros((0,4)); pred_scores=torch.zeros(0)
            pred_labels=torch.zeros(0,dtype=torch.long)

        if has_tm:
            metric.update(  [{"boxes":pred_boxes,"scores":pred_scores,"labels":pred_labels}],
                            [{"boxes":gt_boxes_t,"labels":gt_labels_t}])
            metric50.update([{"boxes":pred_boxes,"scores":pred_scores,"labels":pred_labels}],
                            [{"boxes":gt_boxes_t,"labels":gt_labels_t}])
            metric75.update([{"boxes":pred_boxes,"scores":pred_scores,"labels":pred_labels}],
                            [{"boxes":gt_boxes_t,"labels":gt_labels_t}])

        img_tp = img_fp = img_fn = 0
        matched_gt = set(); matched_det = set()
        order = pred_scores.argsort(descending=True).tolist() if pred_scores.numel()>0 else []
        for j in order:
            cls     = pred_labels[j].item()
            gt_mask = (gt_labels_t==cls).nonzero(as_tuple=True)[0]
            if gt_mask.numel()==0:
                fp_map[cls]+=1; img_fp+=1
                conf_mat[BG_IDX][cls]+=1; matched_det.add(j); continue
            iou = box_iou(pred_boxes[j].unsqueeze(0), gt_boxes_t[gt_mask])[0]
            bv, bk = iou.max(0); gj = gt_mask[bk].item()
            if bv >= args.iou_match and gj not in matched_gt:
                tp_map[cls]+=1; img_tp+=1
                conf_mat[cls][cls]+=1; matched_gt.add(gj); matched_det.add(j)
            else:
                fp_map[cls]+=1; img_fp+=1
                conf_mat[BG_IDX][cls]+=1; matched_det.add(j)

        if pred_boxes.numel()>0 and gt_boxes_t.numel()>0:
            iou_all = box_iou(pred_boxes, gt_boxes_t)
            for j in range(len(det_boxes)):
                if j in matched_det: continue
                bv, gi = iou_all[j].max(0)
                if bv >= args.iou_match:
                    conf_mat[gt_labels_t[gi].item()][pred_labels[j].item()] += 1

        for ii, cls in enumerate(gt_labels_t.tolist()):
            if ii not in matched_gt:
                fn_map[cls]+=1; img_fn+=1; conf_mat[cls][BG_IDX]+=1

        # per-image P/R/F1 for CI
        ip = img_tp/(img_tp+img_fp) if img_tp+img_fp>0 else 0.0
        ir = img_tp/(img_tp+img_fn) if img_tp+img_fn>0 else 0.0
        if1= 2*ip*ir/(ip+ir) if ip+ir>0 else 0.0
        per_img_p.append(ip); per_img_r.append(ir); per_img_f1.append(if1)
        n_evaluated += 1

    del detector; gc.collect(); torch.cuda.empty_cache()

    ap50_map=ap75_map=ap_map={}; r=r50=r75=None
    if has_tm and n_evaluated>0:
        r=metric.compute(); r50=metric50.compute(); r75=metric75.compute()
        def _ex(res):
            c=res.get("classes",torch.tensor([])).tolist()
            v=res.get("map_per_class",torch.tensor([])).tolist()
            return {int(ci):float(vi) for ci,vi in zip(c,v) if float(vi)>=0}
        ap50_map=_ex(r50); ap75_map=_ex(r75); ap_map=_ex(r)

    return (tp_map, fp_map, fn_map, conf_mat,
            ap50_map, ap75_map, ap_map, r50, r75, r, n_evaluated,
            infer_times_ms, per_img_p, per_img_r, per_img_f1)


# ── save ───────────────────────────────────────────────────────────────────────
def save_results(out_dir, name, tp_map, fp_map, fn_map, conf_mat,
                 ap50_map, ap75_map, ap_map, r50, r75, r, has_tm, n_eval, ckpt,
                 infer_times_ms, per_img_p, per_img_r, per_img_f1,
                 params_M, gflops, n_boot, active_classes=None):
    """active_classes: class indices to aggregate P/R/F1/mAP over.
       None = all classes (torchmetrics global mAP).
       Pass EVAL_CLASSES to restrict to A–I + single."""
    os.makedirs(out_dir, exist_ok=True)
    labels_bg = [ID2LABEL[i] for i in range(NUM_PANEL_CLASSES)] + ["background"]
    agg = active_classes if active_classes is not None else range(NUM_PANEL_CLASSES)

    # per_class.csv — always all classes
    with open(os.path.join(out_dir,"per_class.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["Class","Instances","TP","FP","FN","P","R","F1",
                                        "AP@50","AP@75","AP@50:95"])
        w.writeheader()
        for c in range(NUM_PANEL_CLASSES):
            tp=tp_map[c]; fp=fp_map[c]; fn=fn_map[c]
            p=tp/(tp+fp) if tp+fp>0 else 0.
            r_=tp/(tp+fn) if tp+fn>0 else 0.
            f1=2*p*r_/(p+r_) if p+r_>0 else 0.
            fmt=lambda v: f"{v:.4f}" if v>=0 else "-"
            w.writerow({"Class":ID2LABEL[c],"Instances":tp+fn,"TP":tp,"FP":fp,"FN":fn,
                        "P":f"{p:.4f}","R":f"{r_:.4f}","F1":f"{f1:.4f}",
                        "AP@50":fmt(ap50_map.get(c,-1)),"AP@75":fmt(ap75_map.get(c,-1)),
                        "AP@50:95":fmt(ap_map.get(c,-1))})

    # timing stats
    t_arr = np.array(infer_times_ms) if infer_times_ms else np.array([0.0])
    t_mean = float(np.mean(t_arr)); t_std = float(np.std(t_arr))
    fps = round(1000.0 / t_mean, 2) if t_mean > 0 else 0.0

    # bootstrap CI on per-image P / R / F1
    p_mean,  p_lo,  p_hi  = bootstrap_ci(per_img_p,  n_boot)
    r_mean,  r_lo,  r_hi  = bootstrap_ci(per_img_r,  n_boot)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(per_img_f1, n_boot)

    # aggregate TP/FP/FN and P/R/F1 over agg classes
    total_tp = sum(tp_map[c] for c in agg)
    total_fp = sum(fp_map[c] for c in agg)
    total_fn = sum(fn_map[c] for c in agg)
    mp  = total_tp/(total_tp+total_fp) if total_tp+total_fp>0 else 0
    mr  = total_tp/(total_tp+total_fn) if total_tp+total_fn>0 else 0
    mf1 = 2*mp*mr/(mp+mr) if mp+mr>0 else 0

    # mAP: torchmetrics global when all classes; subset average when restricted
    def _cls_ap(ap_dict, c): return round(ap_dict.get(c, -1), 4)
    if active_classes is None:
        map50    = round(r50["map"].item(), 4) if has_tm and r50 else -1
        map75    = round(r75["map"].item(), 4) if has_tm and r75 else -1
        map50_95 = round(r["map"].item(),   4) if has_tm and r   else -1
    else:
        def _subset(ap_dict):
            vals = [ap_dict[c] for c in active_classes if c in ap_dict and ap_dict[c] >= 0]
            return round(sum(vals)/len(vals), 4) if vals else -1
        map50    = _subset(ap50_map)
        map75    = _subset(ap75_map)
        map50_95 = _subset(ap_map)

    record = {
        "model": name, "checkpoint": ckpt, "n_evaluated": n_eval,
        "mean_P":  round(mp,  4), "mean_R":  round(mr,  4), "mean_F1": round(mf1, 4),
        "map50":   map50, "map75": map75, "map50_95": map50_95,
        # 95% bootstrap CI (per-image)
        "P_ci95":  f"{p_mean:.4f} [{p_lo:.4f}, {p_hi:.4f}]",
        "R_ci95":  f"{r_mean:.4f} [{r_lo:.4f}, {r_hi:.4f}]",
        "F1_ci95": f"{f1_mean:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]",
        # efficiency
        "params_M": params_M, "gflops": gflops,
        "infer_ms_mean": round(t_mean, 2), "infer_ms_std": round(t_std, 2),
        "fps": fps,
        "total_TP": total_tp, "total_FP": total_fp, "total_FN": total_fn,
        # per_class record — all classes
        "per_class": {ID2LABEL[c]: {
            "TP": tp_map[c], "FP": fp_map[c], "FN": fn_map[c],
            "AP50":    _cls_ap(ap50_map, c),
            "AP75":    _cls_ap(ap75_map, c),
            "AP50_95": _cls_ap(ap_map,   c),
        } for c in range(NUM_PANEL_CLASSES)}
    }
    with open(os.path.join(out_dir,"results.json"),"w") as f:
        json.dump(record, f, indent=2)
    return record


# ── args / main ────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser("Model comparison — detector only")
    p.add_argument("--test-dir",    default="../MatDetect/all_mydata/test")
    p.add_argument("--val-dir",     default=None,
                   help="Val split dir (default: <parent of test-dir>/val). "
                        "Used only with --combine.")
    p.add_argument("--combine",     action="store_true",
                   help="Combine test + val splits for evaluation")
    p.add_argument("--out-dir",     default="./results_comparison")
    p.add_argument("--conf",        type=float, default=0.3)
    p.add_argument("--iou",         type=float, default=0.5)
    p.add_argument("--iou-match",   type=float, default=0.5)
    p.add_argument("--cuda-device", default="cuda:0")
    p.add_argument("--max-samples", type=int,   default=None)
    p.add_argument("--bootstrap",   type=int,   default=1000,
                   help="bootstrap resamples for 95%% CI (default 1000)")
    p.add_argument("--dabdetr-root", default=None,
                   help="DAB-DETR checkpoint root containing best_path.txt "
                        "(default: ../ModelBench/checkpoints_mydata/dabdetr)")
    p.add_argument("--skip-dabdetr",action="store_true")
    p.add_argument("--skip-yolo",   action="store_true")
    p.add_argument("--eval-classes",action="store_true",
                   help="Restrict aggregate P/R/F1/mAP to A–K + single (EVAL_CLASSES). "
                        "per_class.csv always shows all classes.")
    return p.parse_args()


def load_split_samples(split_dir):
    """Load (img_path, pil, gt_boxes, gt_classes) from a split directory."""
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir)
                   if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    samples = []
    for stem in stems:
        img_path = None
        for ext in IMAGE_EXTS:
            p = os.path.join(images_dir, stem + ext)
            if os.path.exists(p): img_path = p; break
        lbl = os.path.join(labels_dir, stem + ".txt")
        if not img_path or not os.path.exists(lbl): continue
        pil = Image.open(img_path).convert("RGB")
        gb, gc = load_gt(lbl, *pil.size)
        if gb: samples.append((img_path, pil, gb, gc))
    return samples


def main():
    args = get_args()
    args.device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    models = []
    if not args.skip_yolo:   models += find_yolo_models()
    if not args.skip_dabdetr:
        dabdetr_root = args.dabdetr_root or DABDETR_ROOT
        ckpt = resolve_dabdetr(dabdetr_root)
        if ckpt: models.append({"name":"dabdetr_best","type":"dabdetr","ckpt":ckpt})
        else: print(f"[warn] DAB-DETR not found at {dabdetr_root}")

    print(f"Models: {len(models)}")
    for m in models: print(f"  [{m['type']:8}] {m['name']}")

    test_samples = load_split_samples(args.test_dir)
    print(f"Test samples: {len(test_samples)}")

    if args.combine:
        val_dir = args.val_dir or os.path.join(os.path.dirname(args.test_dir), "val")
        val_samples = load_split_samples(val_dir)
        print(f"Val  samples: {len(val_samples)}")
        test_samples = test_samples + val_samples
        print(f"Combined  :   {len(test_samples)}")

    if args.max_samples:
        test_samples = test_samples[:args.max_samples]

    try:
        from torchmetrics.detection import MeanAveragePrecision as MAP
        has_tm = True
    except ImportError:
        has_tm = False; MAP = None

    summary_rows = []
    for model_info in models:
        print(f"\n{'='*60}\n  {model_info['name']}\n{'='*60}")

        print("  Profiling params / GFLOPs...")
        params_M, gflops = profile_model(model_info, args.device)
        print(f"  Params: {params_M:.2f}M   GFLOPs: {gflops:.2f}")

        try:
            (tp,fp,fn,cm,a50,a75,a,r50,r75,r,n,
             times,per_p,per_r,per_f1) = eval_model(
                model_info, test_samples, args, has_tm, MAP)

            rec = save_results(
                os.path.join(args.out_dir, model_info["name"]),
                model_info["name"], tp, fp, fn, cm, a50, a75, a,
                r50, r75, r, has_tm, n, model_info["ckpt"],
                times, per_p, per_r, per_f1,
                params_M, gflops, args.bootstrap,
                active_classes=EVAL_CLASSES if args.eval_classes else None)

            summary_rows.append({
                "model":        rec["model"],
                "type":         model_info["type"],
                "evaluated":    n,
                "params_M":     rec["params_M"],
                "gflops":       rec["gflops"],
                "infer_ms":     f"{rec['infer_ms_mean']:.1f}±{rec['infer_ms_std']:.1f}",
                "fps":          rec["fps"],
                "mean_P":       rec["mean_P"],
                "mean_R":       rec["mean_R"],
                "mean_F1":      rec["mean_F1"],
                "P_ci95":       rec["P_ci95"],
                "R_ci95":       rec["R_ci95"],
                "F1_ci95":      rec["F1_ci95"],
                "mAP@50":       rec["map50"],
                "mAP@75":       rec["map75"],
                "mAP@50:95":    rec["map50_95"],
                "total_TP":     rec["total_TP"],
                "total_FP":     rec["total_FP"],
                "total_FN":     rec["total_FN"],
            })
            print(f"  mAP@50={rec['map50']:.4f}  P={rec['mean_P']:.4f}  "
                  f"R={rec['mean_R']:.4f}  F1={rec['mean_F1']:.4f}  "
                  f"FPS={rec['fps']:.1f}  Params={params_M:.1f}M")
        except Exception as e:
            print(f"[ERROR] {model_info['name']}: {e}")

    if summary_rows:
        sp = os.path.join(args.out_dir,"comparison_summary.csv")
        fields = ["model","type","evaluated",
                  "params_M","gflops","infer_ms","fps",
                  "mean_P","mean_R","mean_F1",
                  "P_ci95","R_ci95","F1_ci95",
                  "mAP@50","mAP@75","mAP@50:95",
                  "total_TP","total_FP","total_FN"]
        with open(sp,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary_rows)

        print(f"\n{'='*60}\n  SUMMARY (sorted by mAP@50)\n{'='*60}")
        print(f"  {'Model':<22} {'mAP@50':>7} {'F1':>7} {'FPS':>7} {'Params(M)':>10} {'GFLOPs':>8}")
        for row in sorted(summary_rows, key=lambda x: -x["mAP@50"]):
            print(f"  {row['model']:<22} {row['mAP@50']:>7.4f} {row['mean_F1']:>7.4f} "
                  f"{row['fps']:>7.1f} {row['params_M']:>10.2f} {row['gflops']:>8.2f}")
        print(f"\n  → {sp}")

if __name__ == "__main__":
    main()
