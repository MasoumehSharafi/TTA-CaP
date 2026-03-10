"""
Online TTA runner (BioVid-friendly) aligned with the paper text.

Aligned behavior:
- At each time t, extract the current target embedding z_t
- Predict with CLIP using fused embedding
- Query:
    - fixed personalized source prototypes
    - positive target cache
    - negative target cache
- Fuse in embedding space:
    z_fuse = z_t + z_src + z_pos - z_neg
- Tri-gate cache update:
    - temporal gate: current predicted label must match majority over last W predictions
    - entropy gate: use current-frame entropy
    - prototype gate: prototype match must agree with current prediction and margin > tau_delta
- Video prediction: average final frame logits over time
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Hashable, Iterable, List, Optional, Tuple

import json
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import clip
from models.vclip import VClip
from utils import (
    build_test_data_loader,
    clip_classifier,
    get_clip_logits,
    get_config_file,
)


# Small utilities

def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _majority(labels: List[int]) -> int:
    cnt = Counter(labels)
    best = max(cnt.values())
    tied = {k for k, v in cnt.items() if v == best}
    if len(tied) == 1:
        return next(iter(tied))
    for y in reversed(labels):
        if y in tied:
            return y
    return labels[-1]


def _unwrap(v):
    if torch.is_tensor(v):
        if v.numel() == 1:
            return v.item()
        return v
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v


def _as_subject_key(x) -> Hashable:
    x = _unwrap(x)
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if int(x) == x:
            return str(int(x))
        return str(x)
    return x


@torch.no_grad()
def compute_metrics_from_lists(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    C = num_classes
    if len(y_true) == 0:
        return {"WAR": 0.0, "UAR": 0.0, "F1_macro": 0.0}

    cm = torch.zeros((C, C), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < C and 0 <= p < C:
            cm[t, p] += 1

    total = cm.sum().item()
    correct = cm.diag().sum().item()
    war = 100.0 * (correct / max(1, total))

    support = cm.sum(dim=1).float()
    pred_count = cm.sum(dim=0).float()

    recall = cm.diag().float() / torch.clamp(support, min=1.0)
    precision = cm.diag().float() / torch.clamp(pred_count, min=1.0)

    uar = 100.0 * recall.mean().item()
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
    f1_macro = 100.0 * f1.mean().item()

    return {"WAR": war, "UAR": uar, "F1_macro": f1_macro}


def entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(p * (p + eps).log()).sum(dim=1)


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def estimate_gflops_clip_visual(clip_model, device: torch.device, input_res: int = 224) -> Optional[float]:
    x = torch.randn(1, 3, input_res, input_res, device=device)

    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        clip_model.eval()
        flops = FlopCountAnalysis(clip_model.visual, x).total()
        return float(flops) / 1e9
    except Exception:
        pass

    try:
        from thop import profile  # type: ignore
        clip_model.eval()
        macs, _params = profile(clip_model.visual, inputs=(x,), verbose=False)
        flops = 2.0 * float(macs)
        return flops / 1e9
    except Exception:
        return None
    
@dataclass
class BufferItem:
    pred: int


class RingBuffer:
    def __init__(self, W: int):
        assert W >= 1
        self.W = W
        self.buf: Deque[BufferItem] = deque(maxlen=W)

    def push(self, item: BufferItem) -> None:
        self.buf.append(item)

    def __len__(self) -> int:
        return len(self.buf)

    def maj_label(self) -> int:
        labels = [b.pred for b in self.buf]
        return _majority(labels)

@dataclass
class CacheEntry:
    z: torch.Tensor      # [1, d] normalized
    entropy: float
    y: torch.Tensor      # [1, C]


class BoundedCache:
    """Per-class bounded cache with entropy-based eviction (keep lowest entropy)."""

    def __init__(self, num_classes: int, capacity_per_class: int, device: torch.device):
        self.C = num_classes
        self.cap = int(capacity_per_class)
        self.device = device
        self.data: Dict[int, List[CacheEntry]] = {c: [] for c in range(self.C)}

    def __len__(self) -> int:
        return sum(len(v) for v in self.data.values())

    def add(self, z: torch.Tensor, cls: int, entropy: float) -> None:
        if self.cap <= 0:
            return

        z = _l2_normalize(z.detach())
        z = _to_device(z, self.device)

        y = F.one_hot(torch.tensor([cls], device=self.device), num_classes=self.C).float()

        entry = CacheEntry(z=z, entropy=float(entropy), y=y)
        bucket = self.data[int(cls)]
        bucket.append(entry)

        bucket.sort(key=lambda e: e.entropy)
        if len(bucket) > self.cap:
            del bucket[self.cap:]

    @torch.no_grad()
    def aggregate_labels(self, q: torch.Tensor, beta: float, k: int) -> torch.Tensor:
        if len(self) == 0:
            return torch.zeros((1, self.C), device=self.device)

        q = _l2_normalize(_to_device(q, self.device))
        keys: List[torch.Tensor] = []
        vals: List[torch.Tensor] = []
        for c in range(self.C):
            for e in self.data[c]:
                keys.append(e.z)
                vals.append(e.y)

        K = torch.cat(keys, dim=0)
        V = torch.cat(vals, dim=0)
        sims = (q @ K.t()).squeeze(0)

        kk = min(int(k), sims.numel())
        topv, topi = torch.topk(sims, k=kk, largest=True)
        w = torch.softmax(topv * float(beta), dim=0)
        out = (w.unsqueeze(1) * V[topi]).sum(dim=0, keepdim=True)
        return out

    @torch.no_grad()
    def aggregate_embed(self, q: torch.Tensor, beta: float, k: int) -> torch.Tensor:
        q = _l2_normalize(_to_device(q, self.device))
        if len(self) == 0:
            return torch.zeros_like(q)

        keys: List[torch.Tensor] = []
        for c in range(self.C):
            for e in self.data[c]:
                keys.append(e.z)

        K = torch.cat(keys, dim=0)
        sims = (q @ K.t()).squeeze(0)

        kk = min(int(k), sims.numel())
        topv, topi = torch.topk(sims, k=kk, largest=True)
        w = torch.softmax(topv * float(beta), dim=0)

        z = (w.unsqueeze(1) * K[topi]).sum(dim=0, keepdim=True)
        z = _l2_normalize(z)
        return z


# Personalized prototypes

@torch.no_grad()
def prototype_scores(q: torch.Tensor, proto_by_class: Dict[int, torch.Tensor], num_classes: int, topk: int, device):
    q = _l2_normalize(_to_device(q, device)).float()
    scores = torch.full((1, num_classes), -1e9, device=device, dtype=q.dtype)

    for c in range(num_classes):
        P = proto_by_class.get(c, None)
        if P is None or (torch.is_tensor(P) and P.numel() == 0):
            continue
        P = _to_device(P, device).float()
        if P.dim() != 2:
            raise ValueError(f"proto_by_class[{c}] must be [N_c, d], got {tuple(P.shape)}")
        P = _l2_normalize(P)

        sims = (q @ P.t()).squeeze(0)
        kk = min(int(topk), sims.numel())
        topv, _ = torch.topk(sims, k=kk, largest=True)
        scores[0, c] = topv.mean()
    return scores


@torch.no_grad()
def prototype_embed(
    q: torch.Tensor,
    proto_by_class: Dict[int, torch.Tensor],
    num_classes: int,
    topk: int,
    device,
    gamma: float = 10.0,
):
    q = _l2_normalize(_to_device(q, device)).float()

    centroids: List[Optional[torch.Tensor]] = [None] * num_classes
    valid = torch.zeros((num_classes,), device=device, dtype=torch.bool)

    for c in range(num_classes):
        P = proto_by_class.get(c, None)
        if P is None or (torch.is_tensor(P) and P.numel() == 0):
            continue
        P = _to_device(P, device).float()
        P = _l2_normalize(P)
        mu = _l2_normalize(P.mean(dim=0, keepdim=True))
        centroids[c] = mu
        valid[c] = True

    s = prototype_scores(q, proto_by_class, num_classes, topk=topk, device=device)
    a = torch.softmax(s * float(gamma), dim=1)

    z_src = torch.zeros_like(q)
    for c in range(num_classes):
        if valid[c]:
            z_src = z_src + a[0, c] * centroids[c]

    z_src = _l2_normalize(z_src)
    return z_src, s


def load_personalized_prototypes(path: str) -> Dict[Hashable, Dict[int, torch.Tensor]]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Prototype file must be a dict (or payload dict containing 'personalized').")
    if "personalized" in obj and isinstance(obj["personalized"], dict):
        obj = obj["personalized"]
    if not isinstance(obj, dict):
        raise ValueError("Prototype file: invalid structure after unwrapping.")

    clean: Dict[Hashable, Dict[int, torch.Tensor]] = {}
    for sid, by_cls in obj.items():
        if not isinstance(by_cls, dict):
            raise ValueError(f"Prototype file: subject '{sid}' must map to dict class->tensor")
        sid_key = _as_subject_key(sid)
        clean[sid_key] = {}
        for c, P in by_cls.items():
            if P is None:
                continue
            if not torch.is_tensor(P):
                raise ValueError(f"Prototype file: subject '{sid}' class '{c}' must be a tensor")
            if P.numel() == 0:
                clean[sid_key][int(c)] = P.float()
                continue
            P = _l2_normalize(P.float())
            clean[sid_key][int(c)] = P
    return clean

def get_arguments():
    p = argparse.ArgumentParser()

    p.add_argument("--config", required=True, help="yaml config directory or file")
    p.add_argument("--head-path", type=str, default=None, help="Path to trained CLIP head checkpoint.")
    p.add_argument("--datasets", type=str, required=True, help="Datasets separated by / (e.g., biovid)")
    p.add_argument("--data-root", type=str, default="./dataset/", help="Path to datasets directory.")
    p.add_argument("--backbone", type=str, choices=["RN50", "ViT-B/16", "ViT-B/32"], required=True)

    p.add_argument("--ft-clip-path", type=str, default=None, help="Path to fine-tuned CLIP checkpoint.")
    p.add_argument("--proto-path", type=str, default=None, help="Path to personalized prototypes .pt")
    p.add_argument("--default-subject", type=str, default="global", help="Fallback subject id when metadata absent.")

    p.add_argument("--save-metrics", type=str, default=None, help="Optional JSON output path.")
    p.add_argument("--save-metrics-txt", type=str, default=None, help="Optional TXT per-subject output path.")

    p.add_argument("--window", type=int, default=3)
    p.add_argument("--proto-topk", type=int, default=5)
    p.add_argument("--cache-topk", type=int, default=5)
    p.add_argument("--temporal", action="store_true", help="Use temporal transformer after CLIP image encoder.")
    p.add_argument("--clip-len", type=int, default=8, help="Temporal window length for BioVid.")
    p.add_argument("--temporal-layers", type=int, default=4, help="Number of temporal transformer layers.")
    p.add_argument("--temporal-heads", type=int, default=8, help="Number of temporal attention heads.")
    p.add_argument("--temporal-ff", type=int, default=2048, help="Temporal transformer FFN hidden dimension.")
    p.add_argument("--temporal-max-len", type=int, default=256, help="Maximum temporal length supported.")
    p.add_argument("--temporal-dropout", type=float, default=0.0, help="Temporal transformer dropout.")

    p.add_argument(
        "--gates",
        type=str,
        default="temp,entropy,proto",
        help="Comma-separated cache-update gates: any subset of {temp,entropy,proto}.",
    )
    p.add_argument(
        "--proto-missing",
        type=str,
        choices=["pass", "block"],
        default="pass",
        help="If proto gate enabled but subject has no prototypes: pass or block.",
    )

    p.add_argument(
        "--fusion-space",
        type=str,
        choices=["logit", "embed"],
        default="embed",
        help="Fusion in logit space or embedding space.",
    )
    p.add_argument(
        "--no-proto-fusion",
        dest="use_proto_fusion",
        action="store_false",
        help="Disable prototype term in fusion.",
    )
    p.set_defaults(use_proto_fusion=True)

    p.add_argument("--proto-gamma", type=float, default=10.0, help="Softmax sharpness for prototype embedding mix.")
    p.add_argument("--input-res", type=int, default=224, help="Input resolution for GFLOPs estimation.")

    return p.parse_args()


# Main TTA loop

@torch.no_grad()
def run_online_tta(
    loader: Iterable,
    clip_model,
    clip_weights: torch.Tensor,  # [D,C]
    proto_db: Optional[Dict[Hashable, Dict[int, torch.Tensor]]],
    head=None,
    *,
    window: int,
    proto_topk: int,
    cache_topk: int,
    tau_H_pos: float,
    tau_H_neg: float,
    tau_delta: float,
    L_pos: int,
    L_neg: int,
    beta: float,
    lambda_src: float,
    lambda_pos: float,
    lambda_neg: float,
    default_subject: Hashable,
    gates: str,
    proto_missing: str,
    fusion_space: str,
    use_proto_fusion: bool,
    proto_gamma: float,
    save_metrics_path: Optional[str],
    save_metrics_txt: Optional[str],
    model_stats: Optional[Dict[str, object]] = None,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    n_frames_total = 0

    clip_weights_model = _to_device(clip_weights, device)
    clip_weights_f32 = clip_weights_model.float()
    C = clip_weights_model.size(1)

    gates_set = {g.strip().lower() for g in gates.split(",") if g.strip()}
    use_temp_gate = ("temp" in gates_set)
    use_entropy_gate = ("entropy" in gates_set)
    use_proto_gate = ("proto" in gates_set)

    print(
        f"[Ablation] update_gates={sorted(list(gates_set))} | "
        f"proto_missing={proto_missing} | fusion_space={fusion_space} | proto_fusion={use_proto_fusion}"
    )

    pos_cache: Dict[Hashable, BoundedCache] = {}
    neg_cache: Dict[Hashable, BoundedCache] = {}
    ring: Dict[Hashable, RingBuffer] = {}

    video_sum: Dict[Tuple[Hashable, Hashable], torch.Tensor] = defaultdict(
        lambda: torch.zeros((1, C), device=device)
    )
    video_cnt: Dict[Tuple[Hashable, Hashable], int] = defaultdict(int)
    video_gt: Dict[Tuple[Hashable, Hashable], int] = {}

    per_subj_ytrue: Dict[Hashable, List[int]] = defaultdict(list)
    per_subj_ypred: Dict[Hashable, List[int]] = defaultdict(list)
    all_ytrue: List[int] = []
    all_ypred: List[int] = []

    frame_counter = 0

    def pass_gate(flag_use: bool, gate_value: int) -> bool:
        return (not flag_use) or bool(gate_value)

    for batch in tqdm(loader, desc="Processed test frames: "):
        n_frames_total += 1

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, target = batch
            meta = {}
        elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
            images, target, meta = batch[0], batch[1], batch[2]
        else:
            raise ValueError("Unexpected batch format. Expected (images, target) or (images, target, meta).")

        target = _to_device(target, device)

        subj_raw = meta.get("subject_id", default_subject) if isinstance(meta, dict) else default_subject
        vid_raw = meta.get("video_id", None) if isinstance(meta, dict) else None
        imp_raw = meta.get("impath", None) if isinstance(meta, dict) else None

        subj = _as_subject_key(subj_raw)
        vid = _unwrap(vid_raw)

        if vid is None:
            vid = _unwrap(imp_raw) if imp_raw is not None else f"frame_{frame_counter}"
        frame_counter += 1

        if subj not in pos_cache:
            pos_cache[subj] = BoundedCache(C, L_pos, device)
            neg_cache[subj] = BoundedCache(C, L_neg, device)
            ring[subj] = RingBuffer(window)

        z_t, logits_base, ent_base, _prob_map, pred_base = get_clip_logits(
            images, clip_model, clip_weights_model, head=head
        )

        z_t = _to_device(z_t, device).float()
        logits_base = _to_device(logits_base, device).float()

        if z_t.dim() != 2 or z_t.size(0) != 1:
            raise RuntimeError("This runner expects batch_size=1 in the test loader (per-frame streaming).")

        z_t = _l2_normalize(z_t)
        p_t = torch.softmax(logits_base, dim=1)
        H_t = float(ent_base.mean().item()) if torch.is_tensor(ent_base) else float(ent_base)
        y_t = int(pred_base)


        # Temporal gate

        ring[subj].push(BufferItem(pred=y_t))
        maj_t = ring[subj].maj_label()
        temp_ok = int(y_t == maj_t)

        # Entropy gate

        conf_ok = int(H_t < tau_H_pos)
        amb_ok = int((tau_H_pos <= H_t) and (H_t < tau_H_neg))


        # Prototype gate

        subj_proto = None
        if proto_db is not None:
            subj_proto = proto_db.get(subj, None)
            if subj_proto is None:
                subj_proto = proto_db.get(str(subj), None)
            if subj_proto is None and len(proto_db) == 1:
                subj_proto = next(iter(proto_db.values()))

        proto_available = subj_proto is not None

        if not proto_available:
            s = torch.zeros((1, C), device=device)
            z_src = torch.zeros_like(z_t)
            if use_proto_gate and proto_missing == "block":
                proto_ok = 0
                margin_ok = 0
            else:
                proto_ok = 1
                margin_ok = 1
        else:
            s = prototype_scores(z_t, subj_proto, C, topk=proto_topk, device=device)
            topv, topi = torch.topk(s, k=min(2, C), dim=1, largest=True)

            c_star = int(topi[0, 0].item())
            delta = float((topv[0, 0] - topv[0, 1]).item()) if topv.shape[1] >= 2 else float("inf")

            proto_ok = int((c_star == y_t) and (delta > tau_delta))
            margin_ok = int(delta > tau_delta)

            z_src, _ = prototype_embed(
                z_t, subj_proto, C, topk=proto_topk, device=device, gamma=proto_gamma
            )


        # Cache updates

        pos_update_ok = (
            pass_gate(use_temp_gate, temp_ok) and
            pass_gate(use_entropy_gate, conf_ok) and
            pass_gate(use_proto_gate, proto_ok)
        )
        neg_update_ok = (
            pass_gate(use_temp_gate, temp_ok) and
            pass_gate(use_entropy_gate, amb_ok) and
            pass_gate(use_proto_gate, margin_ok)
        )

        if pos_update_ok:
            pos_cache[subj].add(z_t, y_t, H_t)
        elif neg_update_ok:
            neg_cache[subj].add(z_t, y_t, H_t)

        # Fusion + final logits
        if fusion_space == "logit":
            l_pos = pos_cache[subj].aggregate_labels(z_t, beta=beta, k=cache_topk)
            l_neg = neg_cache[subj].aggregate_labels(z_t, beta=beta, k=cache_topk)

            if not use_proto_fusion:
                s = torch.zeros((1, C), device=device)

            logits_final = logits_base + (lambda_src * s) + (lambda_pos * l_pos) - (lambda_neg * l_neg)

        else:
            z_pos = pos_cache[subj].aggregate_embed(z_t, beta=beta, k=cache_topk)
            z_neg = neg_cache[subj].aggregate_embed(z_t, beta=beta, k=cache_topk)

            if not use_proto_fusion:
                z_src = torch.zeros_like(z_t)

            z_fused = _l2_normalize(
                z_t + (lambda_src * z_src) + (lambda_pos * z_pos) - (lambda_neg * z_neg)
            )

            if head is not None:
                logits_final = head(z_fused)
            else:
                logits_final = z_fused @ clip_weights_f32
                if hasattr(clip_model, "logit_scale"):
                    logits_final = clip_model.logit_scale.exp().float() * logits_final

        # Video-level aggregation on final logits

        key = (subj, vid)
        video_sum[key] = video_sum[key] + logits_final
        video_cnt[key] += 1

        if key not in video_gt:
            t = _unwrap(target)
            if torch.is_tensor(t):
                t = t.item() if t.numel() == 1 else t[0].item()
            video_gt[key] = int(t)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    elapsed_sec = t_end - t_start
    n_videos_total = max(1, len(video_sum))
    time_per_frame_ms = 1000.0 * elapsed_sec / max(1, n_frames_total)
    time_per_video_ms = 1000.0 * elapsed_sec / n_videos_total

    per_video_records = []
    for key, logit_sum in video_sum.items():
        subj, vid = key
        cnt = max(1, video_cnt[key])
        logit_avg = (logit_sum / float(cnt)).detach()

        pred = int(torch.argmax(logit_avg, dim=1).item())
        gt = int(video_gt[key])

        per_subj_ytrue[subj].append(gt)
        per_subj_ypred[subj].append(pred)
        all_ytrue.append(gt)
        all_ypred.append(pred)

        per_video_records.append({
            "subject_id": str(subj),
            "video_id": str(vid),
            "n_frames": int(cnt),
            "y_true": int(gt),
            "y_pred": int(pred),
            "logits_avg": logit_avg.squeeze(0).float().cpu().tolist(),
        })

    per_subject_metrics = {}
    for subj in per_subj_ytrue.keys():
        m = compute_metrics_from_lists(per_subj_ytrue[subj], per_subj_ypred[subj], num_classes=C)
        per_subject_metrics[str(subj)] = {
            "N_videos": int(len(per_subj_ytrue[subj])),
            "WAR": float(m["WAR"]),
            "UAR": float(m["UAR"]),
            "F1_macro": float(m["F1_macro"]),
        }

    m_all = compute_metrics_from_lists(all_ytrue, all_ypred, num_classes=C)
    overall = {
        "N_videos": int(len(all_ytrue)),
        "WAR": float(m_all["WAR"]),
        "UAR": float(m_all["UAR"]),
        "F1_macro": float(m_all["F1_macro"]),
    }

    if save_metrics_txt is not None:
        lines = []
        lines.append("Per-target-subject (video-level) metrics\n")
        lines.append("subject_id\tN_videos\tWAR\tUAR\tF1_macro\n")
        for subj in sorted(per_subject_metrics.keys(), key=lambda x: int(x) if x.isdigit() else x):
            s_m = per_subject_metrics[subj]
            lines.append(f"{subj}\t{s_m['N_videos']}\t{s_m['WAR']:.4f}\t{s_m['UAR']:.4f}\t{s_m['F1_macro']:.4f}\n")

        lines.append("\nOverall (all subjects)\n")
        lines.append(f"N_videos\t{overall['N_videos']}\n")
        lines.append(f"WAR\t{overall['WAR']:.4f}\n")
        lines.append(f"UAR\t{overall['UAR']:.4f}\n")
        lines.append(f"F1_macro\t{overall['F1_macro']:.4f}\n")

        lines.append("\nRuntime / compute stats\n")
        lines.append(f"n_frames_total\t{n_frames_total}\n")
        lines.append(f"n_videos_total\t{len(video_sum)}\n")
        lines.append(f"total_runtime_sec\t{elapsed_sec:.6f}\n")
        lines.append(f"time_per_frame_ms\t{time_per_frame_ms:.6f}\n")
        lines.append(f"time_per_video_ms\t{time_per_video_ms:.6f}\n")

        if model_stats is not None:
            lines.append("\nModel stats\n")
            lines.append(f"backbone\t{model_stats.get('backbone')}\n")
            lines.append(f"clip_params_total\t{model_stats.get('clip_params_total')}\n")
            lines.append(f"clip_params_trainable\t{model_stats.get('clip_params_trainable')}\n")
            lines.append(f"head_params_total\t{model_stats.get('head_params_total')}\n")
            lines.append(f"head_params_trainable\t{model_stats.get('head_params_trainable')}\n")
            g = model_stats.get("gflops_visual", None)
            lines.append(f"gflops_visual\t{('NA' if g is None else f'{float(g):.4f}')}\n")

        lines.append("\nRun settings\n")
        lines.append(f"update_gates\t{','.join(sorted(list(gates_set)))}\n")
        lines.append(f"proto_missing\t{proto_missing}\n")
        lines.append(f"fusion_space\t{fusion_space}\n")
        lines.append(f"proto_fusion\t{int(use_proto_fusion)}\n")
        lines.append(f"proto_gamma\t{proto_gamma}\n")

        with open(save_metrics_txt, "w") as ftxt:
            ftxt.writelines(lines)
        print(f"[Saved per-subject TXT] {save_metrics_txt}")

    print("\nPer-target-subject (video-level) metrics:")
    for subj in sorted(per_subject_metrics.keys(), key=lambda x: int(x) if x.isdigit() else x):
        s_m = per_subject_metrics[subj]
        print(
            f"  subject {subj}: N_videos={s_m['N_videos']:4d} | "
            f"WAR={s_m['WAR']:.2f} | UAR={s_m['UAR']:.2f} | F1_macro={s_m['F1_macro']:.2f}"
        )

    print(
        f"\nOverall (all subjects) | N_videos={overall['N_videos']} | "
        f"WAR={overall['WAR']:.2f} | UAR={overall['UAR']:.2f} | F1_macro={overall['F1_macro']:.2f}\n"
    )
    print(f"[Runtime] total={elapsed_sec:.3f}s | per_frame={time_per_frame_ms:.3f}ms | per_video={time_per_video_ms:.3f}ms")

    if save_metrics_path is not None:
        payload = {
            "overall": overall,
            "per_subject": per_subject_metrics,
            "per_video": sorted(
                per_video_records,
                key=lambda r: (
                    int(r["subject_id"]) if r["subject_id"].isdigit() else r["subject_id"],
                    r["video_id"],
                ),
            ),
            "settings": {
                "update_gates": sorted(list(gates_set)),
                "proto_missing": proto_missing,
                "fusion_space": fusion_space,
                "proto_fusion": bool(use_proto_fusion),
                "proto_gamma": proto_gamma,
            },
            "runtime": {
                "n_frames_total": n_frames_total,
                "n_videos_total": len(video_sum),
                "total_runtime_sec": elapsed_sec,
                "time_per_frame_ms": time_per_frame_ms,
                "time_per_video_ms": time_per_video_ms,
            },
            "model_stats": model_stats,
        }
        with open(save_metrics_path, "w") as fjson:
            json.dump(payload, fjson, indent=2)
        print(f"[Saved metrics] {save_metrics_path}")

    return float(overall["WAR"])


def main():
    args = get_arguments()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    base_clip_model, preprocess = clip.load(args.backbone, device=device_str)

    if args.ft_clip_path:
        ckpt = torch.load(args.ft_clip_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = base_clip_model.load_state_dict(state, strict=False)
        print(f"[FT-CLIP] Loaded: {args.ft_clip_path}")
        if len(missing) > 0:
            print(f"[FT-CLIP] Missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[FT-CLIP] Unexpected keys: {len(unexpected)}")

    if args.temporal:
        if args.backbone != "ViT-B/32":
            print("[Warning] Current temporal wrapper assumes d_model=512. ViT-B/32 is safest.")

        clip_model = VClip(
            backbone=base_clip_model,
            d_model=512,
            nhead=args.temporal_heads,
            num_layers=args.temporal_layers,
            dim_forward=args.temporal_ff,
            max_len=args.temporal_max_len,
            dropout=args.temporal_dropout,
            freeze_backbone=True,
        ).to(device)
    else:
        clip_model = base_clip_model

    clip_model.eval()

    random.seed(1)
    torch.manual_seed(1)

    proto_db = load_personalized_prototypes(args.proto_path) if args.proto_path else None

    head = None
    if args.head_path:
        ckpt = torch.load(args.head_path, map_location="cpu")
        feat_dim = int(ckpt["feat_dim"])
        num_classes = len(ckpt["classnames"])
        head = torch.nn.Linear(feat_dim, num_classes)
        head.load_state_dict(ckpt["state_dict"])
        head = head.to(device_str)
        head.eval()

    clip_total, clip_train = count_params(clip_model)
    head_total, head_train = (0, 0)
    if head is not None:
        head_total, head_train = count_params(head)

    gflops_visual = estimate_gflops_clip_visual(clip_model, device=device, input_res=args.input_res)

    model_stats = {
        "backbone": args.backbone,
        "clip_params_total": int(clip_total),
        "clip_params_trainable": int(clip_train),
        "head_params_total": int(head_total),
        "head_params_trainable": int(head_train),
        "gflops_visual": (None if gflops_visual is None else float(gflops_visual)),
        "gflops_input_res": int(args.input_res),
    }

    datasets = args.datasets.split("/")
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")
        cfg = get_config_file(args.config, dataset_name)
        print("\nRunning dataset configurations:\n", cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(
            dataset_name,
            args.data_root,
            preprocess,
            temporal=args.temporal,
            clip_len=args.clip_len,
        )
        clip_weights = clip_classifier(classnames, template, clip_model)

        tri = cfg.get("tri_gate", {})
        ent_cfg = tri.get("entropy", {})
        tau_H_pos = float(ent_cfg.get("tau_pos", 0.5))
        tau_H_neg = float(ent_cfg.get("tau_neg", 1.2))
        tau_delta = float(tri.get("tau_delta", 0.05))

        cache_cfg = cfg.get("cache", {})
        L_pos = int(cache_cfg.get("L_pos", 5))
        L_neg = int(cache_cfg.get("L_neg", 5))
        beta = float(cache_cfg.get("beta", 10.0))

        fusion = cfg.get("fusion", {})
        lambda_src = float(fusion.get("lambda_src", 1.0))
        lambda_pos = float(fusion.get("lambda_pos", 1.0))
        lambda_neg = float(fusion.get("lambda_neg", 1.0))

        acc = run_online_tta(
            test_loader,
            clip_model,
            clip_weights,
            proto_db,
            head=head,
            window=args.window,
            proto_topk=args.proto_topk,
            cache_topk=args.cache_topk,
            tau_H_pos=tau_H_pos,
            tau_H_neg=tau_H_neg,
            tau_delta=tau_delta,
            L_pos=L_pos,
            L_neg=L_neg,
            beta=beta,
            lambda_src=lambda_src,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            default_subject=args.default_subject,
            gates=args.gates,
            proto_missing=args.proto_missing,
            fusion_space=args.fusion_space,
            use_proto_fusion=args.use_proto_fusion,
            proto_gamma=args.proto_gamma,
            save_metrics_path=args.save_metrics,
            save_metrics_txt=args.save_metrics_txt,
            model_stats=model_stats,
        )

        print(f"---- Online TTA overall WAR: {acc:.2f}. ----\n")


if __name__ == "__main__":
    main()