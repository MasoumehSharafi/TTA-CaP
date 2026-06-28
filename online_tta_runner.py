from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

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
    load_clip_backbone_checkpoint,
    load_vclip_checkpoint,
    natural_key,
)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def unwrap(value):
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap(value[0])
    return value


def as_key(value) -> str:
    value = unwrap(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    return str(value)


def majority_label(labels: Sequence[int]) -> int:
    if not labels:
        raise ValueError("Cannot compute a majority label from an empty sequence.")
    counts = Counter(labels)
    maximum = max(counts.values())
    tied = {label for label, count in counts.items() if count == maximum}
    for label in reversed(labels):
        if label in tied:
            return label
    return labels[-1]


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    if not y_true:
        return {"WAR": 0.0, "F1_macro": 0.0}

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(y_true, y_pred):
        if 0 <= target < num_classes and 0 <= prediction < num_classes:
            confusion[target, prediction] += 1

    total = int(confusion.sum().item())
    war = 100.0 * float(confusion.diag().sum().item()) / max(1, total)
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    true_positive = confusion.diag().float()
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"WAR": war, "F1_macro": 100.0 * float(f1.mean().item())}


@dataclass
class CacheEntry:
    embedding: torch.Tensor
    entropy: float
    class_index: int


class BoundedCache:
    """Per-class fixed-capacity cache that retains the lowest-entropy entries."""

    def __init__(self, num_classes: int, capacity_per_class: int, device: torch.device) -> None:
        self.num_classes = int(num_classes)
        self.capacity = int(capacity_per_class)
        self.device = device
        self.data: Dict[int, List[CacheEntry]] = {c: [] for c in range(self.num_classes)}

    def __len__(self) -> int:
        return sum(len(entries) for entries in self.data.values())

    def add(self, embedding: torch.Tensor, class_index: int, entropy: float) -> None:
        if self.capacity <= 0:
            return
        class_index = int(class_index)
        if class_index not in self.data:
            raise ValueError(f"Invalid cache class index: {class_index}")
        entry = CacheEntry(
            embedding=l2_normalize(embedding.detach().to(self.device).float()),
            entropy=float(entropy),
            class_index=class_index,
        )
        bucket = self.data[class_index]
        bucket.append(entry)
        bucket.sort(key=lambda item: item.entropy)
        del bucket[self.capacity:]

    @torch.no_grad()
    def retrieve_embedding(self, query: torch.Tensor, class_index: int, topk: int, beta: float) -> torch.Tensor:
        query = l2_normalize(query.to(self.device).float())
        bucket = self.data.get(int(class_index), [])
        if not bucket:
            return torch.zeros_like(query)
        keys = torch.cat([entry.embedding for entry in bucket], dim=0)
        similarities = (query @ keys.t()).squeeze(0)
        count = min(max(1, int(topk)), similarities.numel())
        values, indices = torch.topk(similarities, k=count, largest=True)
        weights = torch.softmax(float(beta) * values, dim=0)
        return l2_normalize((weights.unsqueeze(1) * keys[indices]).sum(dim=0, keepdim=True))

    @torch.no_grad()
    def retrieve_label_scores(self, query: torch.Tensor, topk: int, beta: float) -> torch.Tensor:
        query = l2_normalize(query.to(self.device).float())
        entries = [entry for bucket in self.data.values() for entry in bucket]
        if not entries:
            return torch.zeros((1, self.num_classes), device=self.device)
        keys = torch.cat([entry.embedding for entry in entries], dim=0)
        labels = F.one_hot(
            torch.tensor([entry.class_index for entry in entries], device=self.device),
            num_classes=self.num_classes,
        ).float()
        similarities = (query @ keys.t()).squeeze(0)
        count = min(max(1, int(topk)), similarities.numel())
        values, indices = torch.topk(similarities, k=count, largest=True)
        weights = torch.softmax(float(beta) * values, dim=0)
        return (weights.unsqueeze(1) * labels[indices]).sum(dim=0, keepdim=True)


class PredictionHistory:
    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError("Temporal gate window must be >= 1.")
        self.values: Deque[int] = deque(maxlen=int(length))

    def append(self, prediction: int) -> None:
        self.values.append(int(prediction))

    def majority(self) -> int:
        return majority_label(list(self.values))


@torch.no_grad()
def prototype_scores(
    query: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
    num_classes: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    query = l2_normalize(query.to(device).float())
    scores = torch.full((1, num_classes), -1e9, device=device)
    for class_index in range(num_classes):
        class_prototypes = prototypes.get(class_index)
        if class_prototypes is None or class_prototypes.numel() == 0:
            continue
        class_prototypes = l2_normalize(class_prototypes.to(device).float())
        similarities = (query @ class_prototypes.t()).squeeze(0)
        count = min(max(1, int(topk)), similarities.numel())
        scores[0, class_index] = torch.topk(similarities, k=count, largest=True).values.mean()
    return scores


@torch.no_grad()
def retrieve_source_embedding(
    query: torch.Tensor,
    prototypes: Dict[int, torch.Tensor],
    predicted_class: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    query = l2_normalize(query.to(device).float())
    class_prototypes = prototypes.get(int(predicted_class))
    if class_prototypes is None or class_prototypes.numel() == 0:
        return torch.zeros_like(query)
    class_prototypes = l2_normalize(class_prototypes.to(device).float())
    similarities = (query @ class_prototypes.t()).squeeze(0)
    count = min(max(1, int(topk)), similarities.numel())
    indices = torch.topk(similarities, k=count, largest=True).indices
    return l2_normalize(class_prototypes[indices].mean(dim=0, keepdim=True))


def load_personalized_prototypes(path: str) -> Dict[str, Dict[int, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("personalized"), dict):
        payload = payload["personalized"]
    if not isinstance(payload, dict):
        raise ValueError("Prototype file must contain a subject -> class -> tensor mapping.")

    clean: Dict[str, Dict[int, torch.Tensor]] = {}
    for subject_id, by_class in payload.items():
        if not isinstance(by_class, dict):
            raise ValueError(f"Invalid prototype entry for subject {subject_id!r}.")
        clean[str(subject_id)] = {}
        for class_index, value in by_class.items():
            if not torch.is_tensor(value):
                raise ValueError(f"Prototype subject={subject_id}, class={class_index} is not a tensor.")
            clean[str(subject_id)][int(class_index)] = l2_normalize(value.float()) if value.numel() else value.float()
    return clean


def embedding_prediction(
    embedding: torch.Tensor,
    clip_weights: torch.Tensor,
    logit_scale: torch.Tensor,
) -> Tuple[int, float]:
    logits = logit_scale.float() * (l2_normalize(embedding.float()) @ clip_weights.float())
    probabilities = logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    return int(prediction.item()), float(confidence.item())


def nonzero_embedding(embedding: torch.Tensor, eps: float = 1e-10) -> bool:
    return bool(float(embedding.norm().item()) > eps)


def select_positive_components(
    components: List[Tuple[str, torch.Tensor, int, float]],
    agreement_mode: str,
) -> List[Tuple[str, torch.Tensor, int, float]]:
    if agreement_mode == "none" or len(components) <= 1:
        return components
    counts = Counter(component[2] for component in components)
    majority_class, count = counts.most_common(1)[0]
    if count >= 2:
        return [component for component in components if component[2] == majority_class]
    return [max(components, key=lambda component: component[3])]


@torch.no_grad()
def fuse_embeddings(
    target_embedding: torch.Tensor,
    source_embedding: torch.Tensor,
    positive_embedding: torch.Tensor,
    negative_embedding: torch.Tensor,
    clip_weights: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    weighting_strategy: str,
    agreement_mode: str,
    attention_temperature: float,
    lambda_src: float,
    lambda_pos: float,
    lambda_neg: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    raw_positive = [("target", target_embedding, 1.0)]
    if nonzero_embedding(source_embedding):
        raw_positive.append(("source", source_embedding, float(lambda_src)))
    if nonzero_embedding(positive_embedding):
        raw_positive.append(("positive", positive_embedding, float(lambda_pos)))

    positive_components: List[Tuple[str, torch.Tensor, int, float]] = []
    for name, embedding, base_weight in raw_positive:
        prediction, confidence = embedding_prediction(embedding, clip_weights, logit_scale)
        positive_components.append((name, embedding, prediction, confidence * base_weight))
    positive_components = select_positive_components(positive_components, agreement_mode)

    negative_component = None
    if nonzero_embedding(negative_embedding):
        prediction, confidence = embedding_prediction(negative_embedding, clip_weights, logit_scale)
        negative_component = ("negative", negative_embedding, prediction, confidence * float(lambda_neg))

    names = [component[0] for component in positive_components]
    scores = torch.tensor([component[3] for component in positive_components], device=target_embedding.device)
    if negative_component is not None:
        names.append("negative")
        scores = torch.cat([scores, torch.tensor([negative_component[3]], device=scores.device)])

    if weighting_strategy == "equal":
        weights = torch.ones_like(scores)
        # Preserve the explicit lambda values for equal fusion.
        for index, component in enumerate(positive_components):
            weights[index] = 1.0 if component[0] == "target" else (lambda_src if component[0] == "source" else lambda_pos)
        if negative_component is not None:
            weights[-1] = lambda_neg
    elif weighting_strategy == "confidence":
        weights = scores
    elif weighting_strategy == "attention":
        temperature = max(float(attention_temperature), 1e-6)
        weights = torch.softmax(scores / temperature, dim=0)
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")

    fused = torch.zeros_like(target_embedding)
    details: Dict[str, float] = {}
    for index, component in enumerate(positive_components):
        fused = fused + weights[index] * component[1]
        details[f"weight_{component[0]}"] = float(weights[index].item())
        details[f"confidence_{component[0]}"] = float(component[3])
    if negative_component is not None:
        fused = fused - weights[-1] * negative_component[1]
        details["weight_negative"] = float(weights[-1].item())
        details["confidence_negative"] = float(negative_component[3])
    else:
        details["weight_negative"] = 0.0

    return l2_normalize(fused), details


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


@torch.no_grad()
def estimate_visual_gflops(model, device: torch.device, input_resolution: int) -> Optional[float]:
    visual = model.backbone.visual if hasattr(model, "backbone") else model.visual
    sample = torch.randn(1, 3, input_resolution, input_resolution, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
        return float(FlopCountAnalysis(visual, sample).total()) / 1e9
    except Exception:
        return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTA-CaP online test-time adaptation.")
    parser.add_argument("--config", required=True, help="Config directory or YAML file.")
    parser.add_argument("--datasets", required=True, help="Dataset names separated by '/', e.g. biovid/stressid.")
    parser.add_argument("--data-root", required=True, help="Parent data root or exact dataset root.")
    parser.add_argument("--backbone", required=True, choices=["RN50", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--ft-clip-path", default=None, help="Source-trained CLIP checkpoint.")
    parser.add_argument("--temporal-ckpt-path", default=None, help="Optional full VClip/temporal checkpoint.")
    parser.add_argument("--head-path", default=None, help="Optional frozen linear classifier checkpoint.")
    parser.add_argument("--proto-path", required=True, help="Personalized prototype cache file.")
    parser.add_argument("--save-metrics", default=None, help="JSON output path.")
    parser.add_argument("--save-metrics-txt", default=None, help="Plain-text output path.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--clip-len", type=int, default=8, help="Causal temporal encoder clip length.")
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--temporal-ff", type=int, default=2048)
    parser.add_argument("--temporal-max-len", type=int, default=256)
    parser.add_argument("--temporal-dropout", type=float, default=0.0)

    parser.add_argument("--window", type=int, default=3, help="Tri-gate prediction-history window W.")
    parser.add_argument("--proto-topk", type=int, default=5)
    parser.add_argument("--cache-topk", type=int, default=5)
    parser.add_argument("--gates", default="temp,entropy,proto")
    parser.add_argument("--proto-missing", choices=["pass", "block"], default="block")
    parser.add_argument("--fusion-space", choices=["embed", "logit"], default="embed")
    parser.add_argument("--weighting-strategy", choices=["equal", "confidence", "attention"], default="equal")
    parser.add_argument("--agreement-mode", choices=["none", "majority"], default="none")
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--input-res", type=int, default=224)
    return parser.parse_args()


@torch.no_grad()
def run_online_tta(
    loader: Iterable,
    clip_model,
    clip_weights: torch.Tensor,
    prototype_database: Dict[str, Dict[int, torch.Tensor]],
    *,
    head: Optional[torch.nn.Module],
    window: int,
    proto_topk: int,
    cache_topk: int,
    tau_entropy_positive: float,
    tau_entropy_negative: float,
    tau_prototype_margin: float,
    positive_capacity: int,
    negative_capacity: int,
    cache_beta: float,
    lambda_src: float,
    lambda_pos: float,
    lambda_neg: float,
    gates: str,
    proto_missing: str,
    fusion_space: str,
    weighting_strategy: str,
    agreement_mode: str,
    attention_temperature: float,
    save_metrics_path: Optional[str],
    save_metrics_txt: Optional[str],
    model_stats: Dict[str, object],
) -> float:
    device = clip_weights.device
    num_classes = clip_weights.shape[1]
    logit_scale = clip_model.logit_scale.exp().float() if hasattr(clip_model, "logit_scale") else torch.tensor(100.0, device=device)
    enabled_gates = {gate.strip().lower() for gate in gates.split(",") if gate.strip()}
    unknown_gates = enabled_gates - {"temp", "entropy", "proto"}
    if unknown_gates:
        raise ValueError(f"Unknown gates: {sorted(unknown_gates)}")

    current_stream: Optional[Tuple[str, str]] = None
    completed_streams: set[Tuple[str, str]] = set()
    positive_cache: Optional[BoundedCache] = None
    negative_cache: Optional[BoundedCache] = None
    history: Optional[PredictionHistory] = None

    video_logits: Dict[Tuple[str, str], torch.Tensor] = defaultdict(
        lambda: torch.zeros((1, num_classes), device=device)
    )
    video_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    video_ground_truth: Dict[Tuple[str, str], int] = {}

    gate_counts = Counter()
    per_video_fusion_weights: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    frame_count = 0
    positive_updates = 0
    negative_updates = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for batch in tqdm(loader, desc="Online TTA frames"):
        if not isinstance(batch, (list, tuple)) or len(batch) < 3:
            raise ValueError("The test loader must return (image, target, metadata).")
        images, target, metadata = batch[0], batch[1], batch[2]
        subject_id = as_key(metadata["subject_id"])
        video_id = as_key(metadata["video_id"])
        stream_key = (subject_id, video_id)

        if current_stream != stream_key:
            if stream_key in completed_streams:
                raise RuntimeError(
                    f"Video stream {stream_key} is not contiguous. Keep frames sorted by subject, video, and frame."
                )
            if current_stream is not None:
                completed_streams.add(current_stream)
            current_stream = stream_key
            positive_cache = BoundedCache(num_classes, positive_capacity, device)
            negative_cache = BoundedCache(num_classes, negative_capacity, device)
            history = PredictionHistory(window)

        assert positive_cache is not None and negative_cache is not None and history is not None
        target = target.to(device, non_blocking=True)
        target_embedding, base_logits, entropy_tensor, probabilities, predicted_class = get_clip_logits(
            images, clip_model, clip_weights, device, head=head
        )
        target_embedding = l2_normalize(target_embedding.float())
        entropy = float(entropy_tensor.item())
        least_likely_class = int(probabilities.argmin(dim=1).item())
        frame_count += 1

        history.append(predicted_class)
        temporal_pass = int(predicted_class == history.majority())
        entropy_confident = int(entropy < tau_entropy_positive)
        entropy_ambiguous = int(tau_entropy_positive <= entropy < tau_entropy_negative)
        entropy_admissible = int(entropy < tau_entropy_negative)

        subject_prototypes = prototype_database.get(subject_id)
        prototypes_available = bool(
            subject_prototypes
            and any(tensor.numel() > 0 for tensor in subject_prototypes.values())
        )
        if prototypes_available:
            scores = prototype_scores(
                target_embedding, subject_prototypes, num_classes, proto_topk, device
            )
            values, indices = torch.topk(scores, k=min(2, num_classes), dim=1)
            prototype_class = int(indices[0, 0].item())
            margin = float(values[0, 0].item() - values[0, 1].item()) if num_classes > 1 else float("inf")
            prototype_pass = int(prototype_class == predicted_class and margin > tau_prototype_margin)
            source_embedding = retrieve_source_embedding(
                target_embedding,
                subject_prototypes,
                predicted_class,
                proto_topk,
                device,
            )
        else:
            scores = torch.zeros((1, num_classes), device=device)
            source_embedding = torch.zeros_like(target_embedding)
            prototype_pass = int(proto_missing == "pass")

        # Retrieve before update to prevent the current sample from retrieving itself.
        positive_embedding = positive_cache.retrieve_embedding(
            target_embedding, predicted_class, cache_topk, cache_beta
        )
        negative_embedding = negative_cache.retrieve_embedding(
            target_embedding, predicted_class, cache_topk, cache_beta
        )

        if fusion_space == "embed":
            fused_embedding, fusion_details = fuse_embeddings(
                target_embedding,
                source_embedding,
                positive_embedding,
                negative_embedding,
                clip_weights,
                logit_scale,
                weighting_strategy=weighting_strategy,
                agreement_mode=agreement_mode,
                attention_temperature=attention_temperature,
                lambda_src=lambda_src,
                lambda_pos=lambda_pos,
                lambda_neg=lambda_neg,
            )
            final_logits = head(fused_embedding) if head is not None else logit_scale * (fused_embedding @ clip_weights.float())
            per_video_fusion_weights[stream_key].append(fusion_details)
        else:
            positive_scores = positive_cache.retrieve_label_scores(target_embedding, cache_topk, cache_beta)
            negative_scores = negative_cache.retrieve_label_scores(target_embedding, cache_topk, cache_beta)
            final_logits = base_logits + lambda_src * scores + lambda_pos * positive_scores - lambda_neg * negative_scores

        video_logits[stream_key] += final_logits
        video_counts[stream_key] += 1
        if stream_key not in video_ground_truth:
            video_ground_truth[stream_key] = int(unwrap(target))

        temp_condition = temporal_pass if "temp" in enabled_gates else 1
        proto_condition = prototype_pass if "proto" in enabled_gates else 1
        entropy_condition = entropy_admissible if "entropy" in enabled_gates else 1
        common_condition = bool(temp_condition and proto_condition)

        positive_update = common_condition and (
            bool(entropy_confident) if "entropy" in enabled_gates else True
        )
        negative_update = common_condition and (
            bool(entropy_ambiguous) if "entropy" in enabled_gates else False
        )

        if positive_update:
            positive_cache.add(target_embedding, predicted_class, entropy)
            positive_updates += 1
        elif negative_update:
            # The least-likely class indexes the negative cache partition.
            negative_cache.add(target_embedding, least_likely_class, entropy)
            negative_updates += 1

        gate_counts["temporal"] += temporal_pass
        gate_counts["entropy"] += entropy_admissible
        gate_counts["prototype"] += prototype_pass
        gate_counts["all"] += int(temporal_pass and entropy_admissible and prototype_pass)

    if current_stream is not None:
        completed_streams.add(current_stream)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    per_subject_true: Dict[str, List[int]] = defaultdict(list)
    per_subject_pred: Dict[str, List[int]] = defaultdict(list)
    all_true: List[int] = []
    all_pred: List[int] = []
    per_video_records: List[Dict] = []

    for stream_key in sorted(video_logits, key=lambda key: (natural_key(key[0]), natural_key(key[1]))):
        subject_id, video_id = stream_key
        average_logits = video_logits[stream_key] / max(1, video_counts[stream_key])
        prediction = int(average_logits.argmax(dim=1).item())
        ground_truth = video_ground_truth[stream_key]
        per_subject_true[subject_id].append(ground_truth)
        per_subject_pred[subject_id].append(prediction)
        all_true.append(ground_truth)
        all_pred.append(prediction)

        weight_records = per_video_fusion_weights.get(stream_key, [])
        mean_weights: Dict[str, float] = {}
        if weight_records:
            keys = sorted({key for record in weight_records for key in record})
            for key in keys:
                values = [record[key] for record in weight_records if key in record]
                mean_weights[key] = float(sum(values) / len(values))

        per_video_records.append(
            {
                "subject_id": subject_id,
                "video_id": video_id,
                "n_frames": video_counts[stream_key],
                "y_true": ground_truth,
                "y_pred": prediction,
                "logits_avg": average_logits.squeeze(0).cpu().tolist(),
                "mean_fusion_weights": mean_weights,
            }
        )

    per_subject = {
        subject_id: {
            "N_videos": len(per_subject_true[subject_id]),
            **compute_metrics(per_subject_true[subject_id], per_subject_pred[subject_id], num_classes),
        }
        for subject_id in sorted(per_subject_true, key=natural_key)
    }
    overall = {"N_videos": len(all_true), **compute_metrics(all_true, all_pred, num_classes)}
    pass_rates = {
        key: 100.0 * gate_counts[key] / max(1, frame_count)
        for key in ("temporal", "entropy", "prototype", "all")
    }
    runtime = {
        "n_frames_total": frame_count,
        "n_videos_total": len(video_logits),
        "total_runtime_sec": elapsed,
        "time_per_frame_ms": 1000.0 * elapsed / max(1, frame_count),
        "time_per_video_ms": 1000.0 * elapsed / max(1, len(video_logits)),
    }
    settings = {
        "gates": sorted(enabled_gates),
        "fusion_space": fusion_space,
        "weighting_strategy": weighting_strategy,
        "agreement_mode": agreement_mode,
        "attention_temperature": attention_temperature,
        "positive_updates": positive_updates,
        "negative_updates": negative_updates,
    }

    payload = {
        "overall": overall,
        "per_subject": per_subject,
        "per_video": per_video_records,
        "gate_pass_rates": pass_rates,
        "runtime": runtime,
        "settings": settings,
        "model_stats": model_stats,
    }

    print("\nPer-subject video-level results")
    for subject_id, values in per_subject.items():
        print(
            f"  {subject_id}: N={values['N_videos']} | "
            f"WAR={values['WAR']:.2f} | F1={values['F1_macro']:.2f}"
        )
    print(
        f"Overall: N={overall['N_videos']} | WAR={overall['WAR']:.2f} | "
        f"F1={overall['F1_macro']:.2f}"
    )
    print(
        "Gate pass rates: "
        + ", ".join(f"{key}={value:.2f}%" for key, value in pass_rates.items())
    )

    if save_metrics_path:
        path = Path(save_metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved JSON metrics: {path}")

    if save_metrics_txt:
        path = Path(save_metrics_txt)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["subject_id\tN_videos\tWAR\tF1_macro\n"]
        for subject_id, values in per_subject.items():
            lines.append(
                f"{subject_id}\t{values['N_videos']}\t{values['WAR']:.4f}\t{values['F1_macro']:.4f}\n"
            )
        lines.extend(
            [
                "\nOverall\n",
                f"N_videos\t{overall['N_videos']}\n",
                f"WAR\t{overall['WAR']:.4f}\n",
                f"F1_macro\t{overall['F1_macro']:.4f}\n",
                "\nGate pass rates (%)\n",
            ]
        )
        lines.extend(f"{key}\t{value:.4f}\n" for key, value in pass_rates.items())
        lines.extend(
            [
                "\nRuntime\n",
                f"n_frames_total\t{runtime['n_frames_total']}\n",
                f"n_videos_total\t{runtime['n_videos_total']}\n",
                f"total_runtime_sec\t{runtime['total_runtime_sec']:.6f}\n",
                f"time_per_frame_ms\t{runtime['time_per_frame_ms']:.6f}\n",
                f"time_per_video_ms\t{runtime['time_per_video_ms']:.6f}\n",
            ]
        )
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(lines)
        print(f"Saved TXT metrics: {path}")

    return float(overall["WAR"])


def load_optional_head(path: Optional[str], device: torch.device) -> Optional[torch.nn.Module]:
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Head checkpoint must contain state_dict, feat_dim, and classnames.")
    head = torch.nn.Linear(int(checkpoint["feat_dim"]), len(checkpoint["classnames"]))
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval()


def main() -> None:
    args = parse_arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_clip, preprocess = clip.load(args.backbone, device=str(device), jit=False)
    missing, unexpected = load_clip_backbone_checkpoint(base_clip, args.ft_clip_path)
    if args.ft_clip_path:
        print(f"Loaded CLIP checkpoint: {args.ft_clip_path}")
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")

    if args.temporal:
        embedding_dim = int(getattr(base_clip.visual, "output_dim", 512))
        clip_model = VClip(
            backbone=base_clip,
            d_model=embedding_dim,
            nhead=args.temporal_heads,
            num_layers=args.temporal_layers,
            dim_forward=args.temporal_ff,
            max_len=args.temporal_max_len,
            dropout=args.temporal_dropout,
            freeze_backbone=True,
        ).to(device)
        temporal_checkpoint = args.temporal_ckpt_path or args.ft_clip_path
        missing, unexpected = load_vclip_checkpoint(clip_model, temporal_checkpoint)
        if temporal_checkpoint:
            print(f"Loaded VClip-compatible checkpoint: {temporal_checkpoint}")
            print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        temporal_missing = [key for key in missing if key.startswith("temporal.")]
        if temporal_missing:
            print(
                "[Warning] Temporal encoder parameters are missing; "
                "provide --temporal-ckpt-path for the trained temporal model."
            )
    else:
        clip_model = base_clip
    clip_model.eval()

    prototype_database = load_personalized_prototypes(args.proto_path)
    head = load_optional_head(args.head_path, device)
    total_parameters, trainable_parameters = count_parameters(clip_model)
    model_stats = {
        "backbone": args.backbone,
        "parameters_total": total_parameters,
        "parameters_trainable": trainable_parameters,
        "visual_gflops": estimate_visual_gflops(clip_model, device, args.input_res),
    }

    dataset_names = [name.strip() for name in args.datasets.split("/") if name.strip()]
    for dataset_name in dataset_names:
        config = get_config_file(args.config, dataset_name)
        loader, classnames, templates = build_test_data_loader(
            dataset_name,
            args.data_root,
            preprocess,
            config,
            temporal=args.temporal,
            clip_len=args.clip_len,
            num_workers=args.num_workers,
        )
        clip_weights = clip_classifier(classnames, templates, clip_model, device)

        tri_gate = config.get("tri_gate", {})
        entropy_cfg = tri_gate.get("entropy", {})
        cache_cfg = config.get("cache", {})
        fusion_cfg = config.get("fusion", {})

        print(f"\nRunning {dataset_name} with classes: {classnames}")
        run_online_tta(
            loader,
            clip_model,
            clip_weights,
            prototype_database,
            head=head,
            window=args.window,
            proto_topk=args.proto_topk,
            cache_topk=args.cache_topk,
            tau_entropy_positive=float(entropy_cfg.get("tau_pos", 0.5)),
            tau_entropy_negative=float(entropy_cfg.get("tau_neg", 0.8)),
            tau_prototype_margin=float(tri_gate.get("tau_delta", 0.05)),
            positive_capacity=int(cache_cfg.get("L_pos", 5)),
            negative_capacity=int(cache_cfg.get("L_neg", 4)),
            cache_beta=float(cache_cfg.get("beta", 10.0)),
            lambda_src=float(fusion_cfg.get("lambda_src", 1.0)),
            lambda_pos=float(fusion_cfg.get("lambda_pos", 1.0)),
            lambda_neg=float(fusion_cfg.get("lambda_neg", 1.0)),
            gates=args.gates,
            proto_missing=args.proto_missing,
            fusion_space=args.fusion_space,
            weighting_strategy=args.weighting_strategy,
            agreement_mode=args.agreement_mode,
            attention_temperature=args.attention_temperature,
            save_metrics_path=args.save_metrics,
            save_metrics_txt=args.save_metrics_txt,
            model_stats=model_stats,
        )


if __name__ == "__main__":
    main()
