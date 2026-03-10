import os
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

import open_clip


# -------------------------
# Config / helpers
# -------------------------

CLASS2IDX = {"neutral": 0, "positive": 1}

def l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def list_jpgs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.jpg")] + [p for p in folder.rglob("*.jpeg")] + [p for p in folder.rglob("*.png")])

def parse_source_subject(video_folder_name: str) -> str:
    # e.g. "081617_m_27-BL1-081" -> "081617_m_27"
    return video_folder_name.split("-")[0]

def safe_mean_var(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # X: [N, d]
    mu = X.mean(dim=0)
    var = X.var(dim=0, unbiased=False)
    return mu, var

def shrink_var(var: torch.Tensor, lam: float) -> torch.Tensor:
    # σ^2 <- (1-λ)σ^2 + λ*1
    return (1.0 - lam) * var + lam * torch.ones_like(var)

def diag_frechet_w2(mu1, var1, mu2, var2) -> float:
    # diagonal Gaussian W2^2
    # ||mu1 - mu2||^2 + sum(var1 + var2 - 2*sqrt(var1*var2))
    dmu = (mu1 - mu2).pow(2).sum().item()
    cross = torch.sqrt(torch.clamp(var1 * var2, min=1e-12))
    dvar = (var1 + var2 - 2.0 * cross).sum().item()
    return float(dmu + dvar)

def compute_knn_distances(X: np.ndarray, k: int) -> np.ndarray:
    # distance to k-th nearest neighbor for each point (k>=2)
    nn = NearestNeighbors(n_neighbors=min(k, len(X)))
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    # dists[:,0] is 0 (self); take k-1 index if possible
    idx = min(k - 1, dists.shape[1] - 1)
    return dists[:, idx]

def dbscan_score(labels: np.ndarray) -> Tuple[float, float, int]:
    # validity: low noise + non-degenerate clusters
    n = len(labels)
    noise = (labels == -1).sum()
    noise_rate = noise / max(n, 1)
    clusters = sorted([c for c in set(labels.tolist()) if c != -1])
    num_clusters = len(clusters)

    if num_clusters <= 0:
        return -1e9, noise_rate, num_clusters

    # simple validity term
    validity = (1.0 - noise_rate) * math.log(num_clusters + 1.0)
    return float(validity), float(noise_rate), int(num_clusters)

def bootstrap_ari(X: np.ndarray, eps: float, min_samples: int, B: int, frac: float = 0.8, seed: int = 0) -> float:
    # stability by ARI on intersections of random subsamples
    rng = np.random.RandomState(seed)
    n = len(X)
    if n < max(min_samples + 1, 10):
        return 0.0

    runs = []
    for b in range(B):
        idx = rng.choice(n, size=max(2, int(frac * n)), replace=False)
        Xb = X[idx]
        lab = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(Xb)
        runs.append((idx, lab))

    aris = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            idx_i, lab_i = runs[i]
            idx_j, lab_j = runs[j]
            inter, pos_i, pos_j = np.intersect1d(idx_i, idx_j, return_indices=True)
            if len(inter) < 5:
                continue
            aris.append(adjusted_rand_score(lab_i[pos_i], lab_j[pos_j]))

    if len(aris) == 0:
        return 0.0
    return float(np.mean(aris))

def cluster_medoids(X: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    # X: [N,d] torch, labels: [N] np
    medoids = []
    clusters = sorted([c for c in set(labels.tolist()) if c != -1])
    Xnp = X.cpu().numpy()

    for c in clusters:
        idx = np.where(labels == c)[0]
        Xc = Xnp[idx]
        mu = Xc.mean(axis=0, keepdims=True)  # [1,d]
        # medoid: closest point to centroid in L2
        d = ((Xc - mu) ** 2).sum(axis=1)
        med = idx[int(np.argmin(d))]
        medoids.append(X[med].unsqueeze(0))

    if len(medoids) == 0:
        return torch.empty((0, X.shape[1]), dtype=X.dtype, device=X.device)
    return torch.cat(medoids, dim=0)

@dataclass
class DBSCANSearch:
    min_samples_list: List[int]
    quantiles: List[float]
    B: int
    alpha: float  # weight for stability

def select_dbscan_params(X: torch.Tensor, search: DBSCANSearch, seed: int = 0) -> Tuple[float, int, Dict]:
    """
    Choose (eps, min_samples) by:
      score = validity + alpha * stability
    eps candidates from quantiles of m-NN distances.
    """
    X = l2n(X).float()
    Xnp = X.cpu().numpy()

    best = (-1e18, None, None, None)  # score, eps, m, meta
    for m in search.min_samples_list:
        if len(Xnp) < max(m + 1, 10):
            continue
        kth = compute_knn_distances(Xnp, k=m)
        eps_candidates = [float(np.quantile(kth, q)) for q in search.quantiles]
        # remove duplicates / zeros
        eps_candidates = sorted(set([e for e in eps_candidates if e > 1e-6]))

        for eps in eps_candidates:
            labels = DBSCAN(eps=eps, min_samples=m, metric="euclidean").fit_predict(Xnp)
            validity, noise_rate, num_clusters = dbscan_score(labels)
            if validity < -1e8:
                continue
            stability = bootstrap_ari(Xnp, eps=eps, min_samples=m, B=search.B, seed=seed)
            score = validity + search.alpha * stability

            if score > best[0]:
                best = (score, eps, m, dict(validity=validity, stability=stability,
                                            noise_rate=noise_rate, num_clusters=num_clusters))
    if best[1] is None:
        # fallback: one "cluster" => single medoid by mean
        return 0.5, 5, dict(fallback=True)
    return float(best[1]), int(best[2]), best[3]


# -------------------------
# CLIP encoder
# -------------------------

@torch.no_grad()
def build_clip_encoder(backbone: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    model = model.to(device).eval()
    return model, preprocess

@torch.no_grad()
def encode_images(model, preprocess, paths: List[Path], device: str, batch_size: int = 64) -> torch.Tensor:
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = []
        for p in batch:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, dim=0).to(device)
        f = model.encode_image(x)
        f = l2n(f).cpu()
        feats.append(f)
    return torch.cat(feats, dim=0) if feats else torch.empty((0, model.visual.output_dim), dtype=torch.float32)


# -------------------------
# Main pipeline
# -------------------------

def collect_source_index(source_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Returns: index[subject_id][class_name] = list(paths)
    Uses both train and validation under source_root.
    """
    idx: Dict[str, Dict[str, List[Path]]] = {}

    for split in ["train", "validation"]:
        for cls in CLASS2IDX.keys():
            cls_dir = source_root / split / cls
            if not cls_dir.exists():
                continue

            # inside: video folders
            for vid_dir in cls_dir.iterdir():
                if not vid_dir.is_dir():
                    continue
                subj = parse_source_subject(vid_dir.name)
                idx.setdefault(subj, {}).setdefault(cls, [])
                idx[subj][cls].extend(list_jpgs(vid_dir))

    return idx

def collect_target_index(target_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    target_root = BioVid_Video
    inside: 1..10
    Returns: index[target_id][class_name] = list(paths)
    """
    idx: Dict[str, Dict[str, List[Path]]] = {}
    for subj_dir in sorted([p for p in target_root.iterdir() if p.is_dir()]):
        t_id = subj_dir.name  # "1", "2", ...
        idx.setdefault(t_id, {})
        for cls in CLASS2IDX.keys():
            cls_dir = subj_dir / cls
            if not cls_dir.exists():
                continue
            idx[t_id].setdefault(cls, [])
            idx[t_id][cls].extend(list_jpgs(cls_dir))
    return idx

def subsample(paths: List[Path], max_n: int, seed: int) -> List[Path]:
    if max_n <= 0 or len(paths) <= max_n:
        return paths
    rng = random.Random(seed)
    paths2 = paths.copy()
    rng.shuffle(paths2)
    return paths2[:max_n]

def build_source_prototypes(
    model, preprocess,
    source_index: Dict[str, Dict[str, List[Path]]],
    device: str,
    batch_size: int,
    max_frames_per_subject_class: int,
    search: DBSCANSearch,
    seed: int = 0
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Returns: M[s][c_idx] = Tensor[R, d] (medoids)
    """
    out: Dict[str, Dict[int, torch.Tensor]] = {}
    subjects = sorted(source_index.keys())

    for s in tqdm(subjects, desc="Source prototypes (DBSCAN medoids)"):
        out[s] = {}
        for cls_name, c_idx in CLASS2IDX.items():
            paths = source_index.get(s, {}).get(cls_name, [])
            paths = subsample(paths, max_frames_per_subject_class, seed=seed + hash((s, cls_name)) % 10000)
            if len(paths) == 0:
                out[s][c_idx] = torch.empty((0, model.visual.output_dim), dtype=torch.float32)
                continue

            F = encode_images(model, preprocess, paths, device=device, batch_size=batch_size)
            F = l2n(F).float()

            eps, m, meta = select_dbscan_params(F, search=search, seed=seed)
            if meta.get("fallback", False) or len(F) < 10:
                # fallback: single prototype = closest to mean
                mu = F.mean(dim=0, keepdim=True)
                d = ((F - mu) ** 2).sum(dim=1)
                med = int(torch.argmin(d).item())
                out[s][c_idx] = F[med:med+1]
            else:
                labels = DBSCAN(eps=eps, min_samples=m, metric="euclidean").fit_predict(F.numpy())
                out[s][c_idx] = cluster_medoids(F, labels)

    return out

def build_personalized_cache(
    source_protos: Dict[str, Dict[int, torch.Tensor]],
    target_index: Dict[str, Dict[str, List[Path]]],
    model, preprocess,
    device: str,
    batch_size: int,
    anchor_class: str,
    top_m: int,
    shrink_lam: float,
    cap_K: int,
    seed: int = 0,
    max_calib_frames: int = 2000,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict]:
    """
    Returns:
      personalized[target_id][c_idx] = Tensor[?, d]
      meta with selected sources per target
    """
    # (I) compute source anchor stats from anchor prototypes
    src_stats = {}
    a_idx = CLASS2IDX[anchor_class]
    for s, per_cls in source_protos.items():
        Zs = per_cls.get(a_idx, torch.empty((0, 1)))
        if Zs.numel() == 0 or Zs.shape[0] == 0:
            continue
        mu, var = safe_mean_var(Zs.float())
        var = shrink_var(var, shrink_lam)
        src_stats[s] = (mu, var)

    personalized: Dict[str, Dict[int, torch.Tensor]] = {}
    meta = {"selected_sources": {}}

    # (II) per target: compute target anchor stats from neutral frames, pick top-m sources
    for t_id in tqdm(sorted(target_index.keys()), desc="Personalized cache per target"):
        paths_anchor = target_index[t_id].get(anchor_class, [])
        paths_anchor = subsample(paths_anchor, max_calib_frames, seed=seed + int(t_id) if t_id.isdigit() else seed)
        if len(paths_anchor) == 0:
            raise RuntimeError(f"Target {t_id} has no anchor frames under class '{anchor_class}'")

        Zt = encode_images(model, preprocess, paths_anchor, device=device, batch_size=batch_size)
        Zt = l2n(Zt).float()
        mu_t, var_t = safe_mean_var(Zt)
        var_t = shrink_var(var_t, shrink_lam)

        # rank sources
        dists = []
        for s, (mu_s, var_s) in src_stats.items():
            d = diag_frechet_w2(mu_t, var_t, mu_s, var_s)
            dists.append((d, s))
        dists.sort(key=lambda x: x[0])
        selected = [s for _, s in dists[:top_m]]
        meta["selected_sources"][t_id] = selected

        # build P^{pers}_{t,c} by union of medoids from selected sources
        personalized[t_id] = {}
        for cls_name, c_idx in CLASS2IDX.items():
            pieces = []
            for s in selected:
                Z = source_protos.get(s, {}).get(c_idx, None)
                if Z is None or Z.shape[0] == 0:
                    continue
                pieces.append(Z)
            if len(pieces) == 0:
                personalized[t_id][c_idx] = torch.empty((0, model.visual.output_dim), dtype=torch.float32)
                continue
            P = l2n(torch.cat(pieces, dim=0).float())

            # optional per-class cap K: keep K prototypes closest to target anchor mean
            if cap_K > 0 and P.shape[0] > cap_K:
                sims = (P @ l2n(mu_t).unsqueeze(-1)).squeeze(-1)  # cosine since both l2
                topk = torch.topk(sims, k=cap_K, largest=True).indices
                P = P[topk]

            personalized[t_id][c_idx] = P.cpu()

    return personalized, meta


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--source-root", type=str, required=True, help="Root containing train/ and validation/")
    parser.add_argument("--target-root", type=str, required=True, help="BioVid_Video folder containing 1..10")

    # clip
    parser.add_argument("--backbone", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)

    # prototype extraction
    parser.add_argument("--max-frames-per-subject-class", type=int, default=3000)
    parser.add_argument("--min-samples", type=str, default="5,10,15")
    parser.add_argument("--quantiles", type=str, default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--bootstraps", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.5)

    # personalization
    parser.add_argument("--anchor-class", type=str, default="neutral")
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--shrink-lam", type=float, default=0.05)
    parser.add_argument("--cap-K", type=int, default=0, help="0 disables cap; else keep K prototypes per class")
    parser.add_argument("--max-calib-frames", type=int, default=2000)

    # output
    parser.add_argument("--out-dir", type=str, default="./proto_out")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess = build_clip_encoder(args.backbone, args.pretrained, args.device)

    # 1) index source + target
    source_index = collect_source_index(Path(args.source_root))
    target_index = collect_target_index(Path(args.target_root))

    print(f"[source] subjects: {len(source_index)}")
    print(f"[target] subjects: {len(target_index)}")

    search = DBSCANSearch(
        min_samples_list=[int(x) for x in args.min_samples.split(",")],
        quantiles=[float(x) for x in args.quantiles.split(",")],
        B=args.bootstraps,
        alpha=args.alpha
    )

    # 2) build source prototypes M_{s,c}
    source_protos = build_source_prototypes(
        model, preprocess,
        source_index=source_index,
        device=args.device,
        batch_size=args.batch_size,
        max_frames_per_subject_class=args.max_frames_per_subject_class,
        search=search,
        seed=args.seed
    )

    torch.save(source_protos, out_dir / "source_prototypes.pt")
    print(f"Saved: {out_dir / 'source_prototypes.pt'}")

    # 3) build personalized cache P^{pers}_{t,c}
    personalized, meta = build_personalized_cache(
        source_protos=source_protos,
        target_index=target_index,
        model=model,
        preprocess=preprocess,
        device=args.device,
        batch_size=args.batch_size,
        anchor_class=args.anchor_class,
        top_m=args.top_m,
        shrink_lam=args.shrink_lam,
        cap_K=args.cap_K,
        seed=args.seed,
        max_calib_frames=args.max_calib_frames,
    )

    payload = {
        "personalized": personalized,
        "class2idx": CLASS2IDX,
        "meta": meta,
        "cfg": vars(args),
    }
    torch.save(payload, out_dir / "personalized_prototypes.pt")
    print(f"Saved: {out_dir / 'personalized_prototypes.pt'}")


if __name__ == "__main__":
    main()
