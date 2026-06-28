from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import clip
from utils import (
    get_config_file,
    list_images,
    load_clip_backbone_checkpoint,
    natural_key,
    resolve_dataset_root,
)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def stable_seed(*parts: object, base: int = 0) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(text).hexdigest()[:8]
    return (int(digest, 16) + int(base)) % (2**31 - 1)


def safe_mean_var(features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty [N,D] features, got {tuple(features.shape)}")
    return features.mean(dim=0), features.var(dim=0, unbiased=False)


def shrink_variance(variance: torch.Tensor, lam: float) -> torch.Tensor:
    if not 0.0 <= lam <= 1.0:
        raise ValueError("shrink_lam must be in [0,1].")
    return (1.0 - lam) * variance + lam * torch.ones_like(variance)


def diagonal_frechet_distance(
    mu_a: torch.Tensor,
    var_a: torch.Tensor,
    mu_b: torch.Tensor,
    var_b: torch.Tensor,
) -> float:
    mean_term = (mu_a - mu_b).pow(2).sum()
    cross = torch.sqrt((var_a * var_b).clamp_min(1e-12))
    covariance_term = (var_a + var_b - 2.0 * cross).sum()
    return float((mean_term + covariance_term).item())


@dataclass(frozen=True)
class DBSCANSearch:
    min_samples: Sequence[int]
    quantiles: Sequence[float]
    bootstraps: int = 6
    bootstrap_fraction: float = 0.8
    stability_weight: float = 0.5
    max_noise_rate: float = 0.95


def kth_neighbor_distances(features: np.ndarray, k: int) -> np.ndarray:
    neighbors = min(max(2, int(k)), len(features))
    model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(features)
    distances, _ = model.kneighbors(features)
    return distances[:, min(neighbors - 1, k - 1)]


def clustering_validity(labels: np.ndarray, max_noise_rate: float) -> Tuple[float, float, int]:
    noise_rate = float(np.mean(labels == -1))
    clusters = [label for label in np.unique(labels) if label != -1]
    num_clusters = len(clusters)
    if num_clusters == 0 or noise_rate > max_noise_rate:
        return -1e9, noise_rate, num_clusters
    # Rewards retained samples and multiple dense modes without favoring fragmentation too strongly.
    score = (1.0 - noise_rate) * math.log1p(num_clusters)
    return float(score), noise_rate, num_clusters


def bootstrap_stability(
    features: np.ndarray,
    eps: float,
    min_samples: int,
    bootstraps: int,
    fraction: float,
    seed: int,
) -> float:
    if bootstraps < 2 or len(features) < max(min_samples + 1, 10):
        return 0.0

    rng = np.random.RandomState(seed)
    runs: List[Tuple[np.ndarray, np.ndarray]] = []
    sample_size = max(2, int(round(fraction * len(features))))
    for _ in range(bootstraps):
        indices = np.sort(rng.choice(len(features), size=sample_size, replace=False))
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features[indices])
        runs.append((indices, labels))

    values: List[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            idx_i, labels_i = runs[i]
            idx_j, labels_j = runs[j]
            overlap, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(overlap) >= 5:
                values.append(adjusted_rand_score(labels_i[pos_i], labels_j[pos_j]))
    return float(np.mean(values)) if values else 0.0


def select_dbscan_parameters(
    features: torch.Tensor,
    search: DBSCANSearch,
    seed: int,
) -> Tuple[Optional[float], Optional[int], Dict[str, float]]:
    normalized = l2_normalize(features.float()).cpu().numpy()
    best_score = -float("inf")
    best: Tuple[Optional[float], Optional[int], Dict[str, float]] = (None, None, {"fallback": 1.0})

    for min_samples in search.min_samples:
        if len(normalized) < max(min_samples + 1, 10):
            continue
        kth = kth_neighbor_distances(normalized, min_samples)
        candidates = sorted({float(np.quantile(kth, q)) for q in search.quantiles if 0.0 < q < 1.0})
        for eps in candidates:
            if eps <= 1e-8:
                continue
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(normalized)
            validity, noise_rate, num_clusters = clustering_validity(labels, search.max_noise_rate)
            if validity < -1e8:
                continue
            stability = bootstrap_stability(
                normalized,
                eps,
                min_samples,
                search.bootstraps,
                search.bootstrap_fraction,
                seed,
            )
            score = validity + search.stability_weight * stability
            if score > best_score:
                best_score = score
                best = (
                    eps,
                    min_samples,
                    {
                        "fallback": 0.0,
                        "score": float(score),
                        "validity": float(validity),
                        "stability": float(stability),
                        "noise_rate": float(noise_rate),
                        "num_clusters": float(num_clusters),
                    },
                )
    return best


def closest_to_mean(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    index = int(((features - mean) ** 2).sum(dim=1).argmin().item())
    return features[index:index + 1]


def cluster_medoids(features: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    medoids: List[torch.Tensor] = []
    numpy_features = features.cpu().numpy()
    for label in sorted(label for label in np.unique(labels) if label != -1):
        indices = np.where(labels == label)[0]
        cluster = numpy_features[indices]
        center = cluster.mean(axis=0, keepdims=True)
        local_index = int(np.argmin(((cluster - center) ** 2).sum(axis=1)))
        medoids.append(features[int(indices[local_index])].unsqueeze(0))
    return torch.cat(medoids, dim=0) if medoids else closest_to_mean(features)


def parse_source_subject(video_name: str, pattern: str) -> str:
    match = re.search(pattern, video_name)
    if match is None:
        raise ValueError(
            f"Cannot extract a source subject from video folder '{video_name}' with regex '{pattern}'."
        )
    return str(match.group(1) if match.groups() else match.group(0))


def collect_source_index(dataset_root: Path, dataset_cfg: Dict) -> Dict[str, Dict[int, List[Path]]]:
    class_folders = list(dataset_cfg["class_folders"])
    splits = list(dataset_cfg.get("source_splits", ["train", "validation"]))
    subject_regex = str(dataset_cfg.get("source_subject_regex", r"^([^-]+)"))
    index: Dict[str, Dict[int, List[Path]]] = {}

    for split in splits:
        for class_index, class_folder in enumerate(class_folders):
            class_root = dataset_root / split / class_folder
            if not class_root.exists():
                continue
            for image_path in list_images(class_root):
                relative = image_path.relative_to(class_root)
                video_name = relative.parts[0] if relative.parts else image_path.parent.name
                subject_id = parse_source_subject(video_name, subject_regex)
                index.setdefault(subject_id, {}).setdefault(class_index, []).append(image_path)

    if not index:
        expected = dataset_root / splits[0] / class_folders[0] / "<video>" / "<frame>.jpg"
        raise RuntimeError(f"No source frames found. Expected a layout similar to: {expected}")
    return index


def collect_target_index(dataset_root: Path, dataset_cfg: Dict) -> Dict[str, List[Path]]:
    target_subdir = str(dataset_cfg.get("target_subdir", "")).strip()
    target_root = dataset_root / target_subdir if target_subdir else dataset_root
    source_splits = set(dataset_cfg.get("source_splits", ["train", "validation"]))
    source_splits.update(dataset_cfg.get("exclude_target_dirs", []))
    pattern = re.compile(str(dataset_cfg.get("target_subject_regex", r".+")))

    index: Dict[str, List[Path]] = {}
    subject_dirs = [
        path for path in target_root.iterdir()
        if path.is_dir()
        and path.name not in source_splits
        and not path.name.startswith(".")
        and pattern.fullmatch(path.name)
    ]
    for subject_dir in sorted(subject_dirs, key=lambda p: natural_key(p.name)):
        paths = list_images(subject_dir)
        if paths:
            index[subject_dir.name] = paths
    if not index:
        raise RuntimeError(f"No target-subject frames found under: {target_root}")
    return index


def subsample(paths: Sequence[Path], max_count: int, seed: int) -> List[Path]:
    values = list(paths)
    if max_count <= 0 or len(values) <= max_count:
        return values
    rng = random.Random(seed)
    rng.shuffle(values)
    return sorted(values[:max_count], key=lambda p: natural_key(p.as_posix()))


@torch.no_grad()
def encode_images(
    model,
    preprocess,
    paths: Sequence[Path],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    batches: List[torch.Tensor] = []
    for start in range(0, len(paths), batch_size):
        images: List[torch.Tensor] = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        batch = torch.stack(images, dim=0).to(device, non_blocking=True)
        features = l2_normalize(model.encode_image(batch).float()).cpu()
        batches.append(features)
    output_dim = int(getattr(model.visual, "output_dim", 512))
    return torch.cat(batches, dim=0) if batches else torch.empty((0, output_dim), dtype=torch.float32)


def build_source_prototypes_and_statistics(
    model,
    preprocess,
    source_index: Dict[str, Dict[int, List[Path]]],
    num_classes: int,
    device: torch.device,
    batch_size: int,
    max_frames_per_subject_class: int,
    search: DBSCANSearch,
    shrink_lam: float,
    seed: int,
):
    prototypes: Dict[str, Dict[int, torch.Tensor]] = {}
    statistics: Dict[str, Dict[str, torch.Tensor]] = {}
    clustering_meta: Dict[str, Dict[int, Dict[str, float]]] = {}
    output_dim = int(getattr(model.visual, "output_dim", 512))

    for subject_id in tqdm(sorted(source_index, key=natural_key), desc="Source subjects"):
        prototypes[subject_id] = {}
        clustering_meta[subject_id] = {}
        all_subject_features: List[torch.Tensor] = []

        for class_index in range(num_classes):
            paths = subsample(
                source_index.get(subject_id, {}).get(class_index, []),
                max_frames_per_subject_class,
                stable_seed(subject_id, class_index, base=seed),
            )
            if not paths:
                prototypes[subject_id][class_index] = torch.empty((0, output_dim), dtype=torch.float32)
                clustering_meta[subject_id][class_index] = {"empty": 1.0}
                continue

            features = encode_images(model, preprocess, paths, device, batch_size)
            features = l2_normalize(features.float())
            all_subject_features.append(features)

            eps, min_samples, meta = select_dbscan_parameters(
                features,
                search,
                stable_seed(subject_id, class_index, "dbscan", base=seed),
            )
            if eps is None or min_samples is None or features.shape[0] < 10:
                class_prototypes = closest_to_mean(features)
                meta = {**meta, "fallback": 1.0}
            else:
                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features.numpy())
                class_prototypes = cluster_medoids(features, labels)
                meta = {**meta, "eps": float(eps), "min_samples": float(min_samples)}

            prototypes[subject_id][class_index] = l2_normalize(class_prototypes).cpu()
            clustering_meta[subject_id][class_index] = meta

        if not all_subject_features:
            raise RuntimeError(f"Source subject '{subject_id}' has no usable frames.")
        subject_features = torch.cat(all_subject_features, dim=0)
        mean, variance = safe_mean_var(subject_features)
        statistics[subject_id] = {
            "mean": mean.cpu(),
            "variance": shrink_variance(variance, shrink_lam).cpu(),
            "num_features": torch.tensor(subject_features.shape[0]),
        }

    return prototypes, statistics, clustering_meta


def build_personalized_caches(
    source_prototypes: Dict[str, Dict[int, torch.Tensor]],
    source_statistics: Dict[str, Dict[str, torch.Tensor]],
    target_index: Dict[str, List[Path]],
    model,
    preprocess,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    top_m: int,
    shrink_lam: float,
    cap_per_class: int,
    max_target_frames: int,
    seed: int,
):
    personalized: Dict[str, Dict[int, torch.Tensor]] = {}
    target_statistics: Dict[str, Dict[str, torch.Tensor]] = {}
    selected_sources: Dict[str, List[str]] = {}
    source_distances: Dict[str, List[Tuple[str, float]]] = {}
    output_dim = int(getattr(model.visual, "output_dim", 512))

    for target_id in tqdm(sorted(target_index, key=natural_key), desc="Target personalization"):
        target_paths = subsample(
            target_index[target_id],
            max_target_frames,
            stable_seed(target_id, "target", base=seed),
        )
        target_features = encode_images(model, preprocess, target_paths, device, batch_size)
        target_features = l2_normalize(target_features.float())
        mean_t, variance_t = safe_mean_var(target_features)
        variance_t = shrink_variance(variance_t, shrink_lam)
        target_statistics[target_id] = {
            "mean": mean_t.cpu(),
            "variance": variance_t.cpu(),
            "num_features": torch.tensor(target_features.shape[0]),
        }

        ranked: List[Tuple[float, str]] = []
        for source_id, stats in source_statistics.items():
            distance = diagonal_frechet_distance(
                mean_t,
                variance_t,
                stats["mean"].float(),
                stats["variance"].float(),
            )
            ranked.append((distance, source_id))
        ranked.sort(key=lambda pair: pair[0])
        selected = [source_id for _, source_id in ranked[:max(1, top_m)]]
        selected_sources[target_id] = selected
        source_distances[target_id] = [(source_id, float(distance)) for distance, source_id in ranked]

        personalized[target_id] = {}
        normalized_target_mean = l2_normalize(mean_t.unsqueeze(0)).squeeze(0)
        for class_index in range(num_classes):
            pieces = [
                source_prototypes[source_id][class_index]
                for source_id in selected
                if class_index in source_prototypes[source_id]
                and source_prototypes[source_id][class_index].numel() > 0
            ]
            if not pieces:
                personalized[target_id][class_index] = torch.empty((0, output_dim), dtype=torch.float32)
                continue
            prototypes = l2_normalize(torch.cat(pieces, dim=0).float())
            if cap_per_class > 0 and prototypes.shape[0] > cap_per_class:
                similarities = prototypes @ normalized_target_mean
                keep = torch.topk(similarities, k=cap_per_class, largest=True).indices
                prototypes = prototypes[keep]
            personalized[target_id][class_index] = prototypes.cpu()

    meta = {
        "selected_sources": selected_sources,
        "source_distances": source_distances,
        "target_statistics": target_statistics,
    }
    return personalized, meta


def parse_number_list(text: str, cast):
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TTA-CaP personalized static caches.")
    parser.add_argument("--config", required=True, help="Config directory or dataset YAML file.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. biovid or stressid.")
    parser.add_argument("--source-root", required=True, help="Parent data root or exact dataset root.")
    parser.add_argument("--target-root", default=None, help="Optional separate parent/exact target root.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backbone", default="ViT-B/32", choices=["RN50", "ViT-B/16", "ViT-B/32"])
    parser.add_argument("--ft-clip-path", default=None, help="Source-trained CLIP checkpoint used by online TTA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--cap-per-class", type=int, default=0, help="0 keeps all selected prototypes.")
    parser.add_argument("--max-source-frames-per-subject-class", type=int, default=0, help="0 uses all source frames in each subject-class subset.")
    parser.add_argument("--max-target-frames", type=int, default=0, help="0 uses all unlabeled target frames.")
    parser.add_argument("--shrink-lam", type=float, default=0.05)
    parser.add_argument("--min-samples", default="5,10,15")
    parser.add_argument("--eps-quantiles", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--bootstraps", type=int, default=6)
    parser.add_argument("--bootstrap-fraction", type=float, default=0.8)
    parser.add_argument("--stability-weight", type=float, default=0.5)
    parser.add_argument("--max-noise-rate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = get_config_file(args.config, args.dataset)
    dataset_cfg = config.get("dataset", {})
    class_folders = list(dataset_cfg.get("class_folders", []))
    class_names = list(dataset_cfg.get("class_names", class_folders))
    if not class_folders or len(class_folders) != len(class_names):
        raise ValueError("Config must define equal-length dataset.class_folders and dataset.class_names.")

    source_root = resolve_dataset_root(args.source_root, dataset_cfg)
    target_root = resolve_dataset_root(args.target_root or args.source_root, dataset_cfg)

    model, preprocess = clip.load(args.backbone, device=str(device), jit=False)
    missing, unexpected = load_clip_backbone_checkpoint(model, args.ft_clip_path)
    if args.ft_clip_path:
        print(f"Loaded source CLIP checkpoint: {args.ft_clip_path}")
        print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    source_index = collect_source_index(source_root, dataset_cfg)
    target_index = collect_target_index(target_root, dataset_cfg)
    print(f"Source subjects: {len(source_index)}")
    print(f"Target subjects: {len(target_index)}")

    search = DBSCANSearch(
        min_samples=parse_number_list(args.min_samples, int),
        quantiles=parse_number_list(args.eps_quantiles, float),
        bootstraps=args.bootstraps,
        bootstrap_fraction=args.bootstrap_fraction,
        stability_weight=args.stability_weight,
        max_noise_rate=args.max_noise_rate,
    )

    source_prototypes, source_statistics, clustering_meta = build_source_prototypes_and_statistics(
        model=model,
        preprocess=preprocess,
        source_index=source_index,
        num_classes=len(class_names),
        device=device,
        batch_size=args.batch_size,
        max_frames_per_subject_class=args.max_source_frames_per_subject_class,
        search=search,
        shrink_lam=args.shrink_lam,
        seed=args.seed,
    )

    personalized, personalization_meta = build_personalized_caches(
        source_prototypes=source_prototypes,
        source_statistics=source_statistics,
        target_index=target_index,
        model=model,
        preprocess=preprocess,
        num_classes=len(class_names),
        device=device,
        batch_size=args.batch_size,
        top_m=args.top_m,
        shrink_lam=args.shrink_lam,
        cap_per_class=args.cap_per_class,
        max_target_frames=args.max_target_frames,
        seed=args.seed,
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_prototypes": source_prototypes,
            "source_statistics": source_statistics,
            "clustering_meta": clustering_meta,
            "class_folders": class_folders,
            "class_names": class_names,
            "config": vars(args),
        },
        output_dir / "source_prototypes.pt",
    )
    torch.save(
        {
            "personalized": personalized,
            "class_folders": class_folders,
            "class_names": class_names,
            "meta": personalization_meta,
            "config": vars(args),
        },
        output_dir / "personalized_prototypes.pt",
    )

    summary = {
        "dataset": args.dataset,
        "source_subjects": len(source_index),
        "target_subjects": len(target_index),
        "top_m": args.top_m,
        "selected_sources": personalization_meta["selected_sources"],
    }
    with (output_dir / "cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved source cache data to: {output_dir / 'source_prototypes.pt'}")
    print(f"Saved personalized caches to: {output_dir / 'personalized_prototypes.pt'}")


if __name__ == "__main__":
    main()
