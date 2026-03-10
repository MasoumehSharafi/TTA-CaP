import os
import yaml
import torch
import math
import numpy as np
import clip
from pathlib import Path
from collections import defaultdict

from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class BioVidTestWithMeta(torch.utils.data.Dataset):
    """
    Wraps a frame-level Datum dataset and returns either:
      - single image tensor [C,H,W]              when temporal=False
      - temporal clip tensor [T,C,H,W]           when temporal=True

    meta contains:
      - subject_id
      - video_id
      - impath
      - frame_idx_in_video
      - clip_len
    """

    def __init__(self, data_source, tfm=None, temporal=False, clip_len=8):
        self.data_source = data_source
        self.tfm = tfm
        self.temporal = temporal
        self.clip_len = int(clip_len)
        self._to_tensor = transforms.ToTensor()

        # build frame groups by video so each frame can retrieve neighboring frames
        self.video_to_indices = defaultdict(list)
        self.video_to_paths = defaultdict(list)

        for idx, item in enumerate(self.data_source):
            impath = item.impath
            subject_id, video_id = self._parse_subject_and_video(impath)
            key = (str(subject_id), str(video_id))
            self.video_to_indices[key].append(idx)
            self.video_to_paths[key].append(impath)

        # sort paths within each video
        for key in self.video_to_paths:
            pairs = sorted(
                zip(self.video_to_paths[key], self.video_to_indices[key]),
                key=lambda x: x[0]
            )
            self.video_to_paths[key] = [p for p, _ in pairs]
            self.video_to_indices[key] = [i for _, i in pairs]

        # map global dataset idx -> position inside its video
        self.idx_to_video_info = {}
        for key in self.video_to_indices:
            for pos, idx in enumerate(self.video_to_indices[key]):
                self.idx_to_video_info[idx] = {
                    "video_key": key,
                    "pos_in_video": pos,
                }

    def __len__(self):
        return len(self.data_source)

    @staticmethod
    def _parse_subject_and_video(impath):
        p = Path(impath)
        parts = p.parts

        subject_id = "global"
        video_id = "unknown"

        for cname in ["neutral", "pain"]:
            if cname in parts:
                i = parts.index(cname)
                if i - 1 >= 0:
                    subject_id = parts[i - 1]
                if i + 1 < len(parts):
                    video_id = parts[i + 1]
                break

        return subject_id, video_id

    def _load_one_image(self, impath):
        img = Image.open(impath).convert("RGB")
        if self.tfm is not None:
            img = self.tfm(img)
        else:
            img = self._to_tensor(img)
        return img

    def _sample_clip_paths(self, video_paths, center_pos):
        """
        Centered temporal window with edge replication.
        """
        if self.clip_len <= 1:
            return [video_paths[center_pos]]

        half = self.clip_len // 2
        if self.clip_len % 2 == 0:
            offsets = list(range(-half, half))
        else:
            offsets = list(range(-half, half + 1))

        clip_paths = []
        n = len(video_paths)
        for off in offsets:
            j = center_pos + off
            j = max(0, min(n - 1, j))
            clip_paths.append(video_paths[j])

        if len(clip_paths) != self.clip_len:
            while len(clip_paths) < self.clip_len:
                clip_paths.append(clip_paths[-1])
            clip_paths = clip_paths[:self.clip_len]

        return clip_paths

    def __getitem__(self, idx):
        item = self.data_source[idx]
        impath = item.impath

        subject_id, video_id = self._parse_subject_and_video(impath)
        video_key = (str(subject_id), str(video_id))

        target = torch.tensor(item.label, dtype=torch.long)

        if not self.temporal:
            img = self._load_one_image(impath)
            meta = {
                "subject_id": int(item.domain) if item.domain is not None else "global",
                "video_id": video_id,
                "impath": impath,
                "frame_idx_in_video": self.idx_to_video_info[idx]["pos_in_video"],
                "clip_len": 1,
            }
            return img, target, meta

        pos_in_video = self.idx_to_video_info[idx]["pos_in_video"]
        video_paths = self.video_to_paths[video_key]
        clip_paths = self._sample_clip_paths(video_paths, pos_in_video)
        clip = torch.stack([self._load_one_image(p) for p in clip_paths], dim=0)   # [T,C,H,W]

        meta = {
            "subject_id": int(item.domain) if item.domain is not None else "global",
            "video_id": video_id,
            "impath": impath,
            "frame_idx_in_video": pos_in_video,
            "clip_len": self.clip_len,
        }
        return clip, target, meta


def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(1))
    return float(loss / max_entropy)


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def get_clip_logits(images, clip_model, clip_weights, head=None):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if head is not None:
            clip_logits = head(image_features.float())
        else:
            clip_logits = 100.0 * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:max(1, int(batch_entropy.size(0) * 0.1))]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def get_ood_preprocess():
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    elif dataset_name == "biovid":
        config_name = "biovid.yaml"
    else:
        config_name = f"{dataset_name}.yaml"

    config_file = os.path.join(config_path, config_name)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, temporal=False, clip_len=8):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=8, shuffle=True)

    elif dataset_name in ['A', 'V', 'R', 'S']:
        preprocess = get_ood_preprocess()
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101', 'oxford_flowers',
                          'oxford_pets', 'stanford_cars', 'sun397', 'ucf101']:
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ["biovid"]:
        dataset = build_dataset("biovid", root_path)

        wrapped = BioVidTestWithMeta(
            dataset.test,
            tfm=preprocess,
            temporal=temporal,
            clip_len=clip_len,
        )

        test_loader = torch.utils.data.DataLoader(
            wrapped,
            batch_size=1,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )

    else:
        raise ValueError("Dataset is not from the chosen list")

    return test_loader, dataset.classnames, dataset.template