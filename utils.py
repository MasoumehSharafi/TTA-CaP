import os
import yaml
import torch
import math
import numpy as np
import clip
from pathlib import Path

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


# -------------------------
# NEW: BioVid wrapper that returns meta
# -------------------------

class BioVidTestWithMeta(torch.utils.data.Dataset):
    """
    Wraps a Datum-style dataset list (dataset.test) and returns:
      (img_tensor, target_tensor, meta_dict)

    meta_dict contains:
      - subject_id: int (from Datum.domain)
      - video_id: str (folder name right after neutral/pain)
      - impath: str
    """
    def __init__(self, data_source, tfm=None):
        self.data_source = data_source
        self.tfm = tfm
        self._to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]  # Datum with .impath, .label, .domain, .classname

        impath = item.impath
        img = Image.open(impath).convert("RGB")

        if self.tfm is not None:
            img = self.tfm(img)
        else:
            img = self._to_tensor(img)

        # Parse video_id from path: .../<subject>/<class>/<video_folder>/.../frame.jpg
        p = Path(impath)
        parts = p.parts
        video_id = "unknown"
        for cname in ["neutral", "pain"]:
            if cname in parts:
                i = parts.index(cname)
                if i + 1 < len(parts):
                    video_id = parts[i + 1]
                break

        meta = {
            "subject_id": int(item.domain) if item.domain is not None else "global",
            "video_id": video_id,
            "impath": impath,
        }

        target = torch.tensor(item.label, dtype=torch.long)
        return img, target, meta


# -------------------------
# Existing utilities
# -------------------------

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
            # Use learned head instead of zero-shot text weights
            clip_logits = head(image_features.float())
        else:
            clip_logits = 100. * image_features @ clip_weights

        if image_features.size(0) > 1:
            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
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


def build_test_data_loader(dataset_name, root_path, preprocess):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        # For online/streaming TTA, shuffle=False is usually better,
        # but I keep your original behavior here.
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

        # IMPORTANT: For your online_tta_runner.py we need meta (subject_id, video_id).
        wrapped = BioVidTestWithMeta(dataset.test, tfm=preprocess)

        test_loader = torch.utils.data.DataLoader(
            wrapped,
            batch_size=1,
            num_workers=8,
            shuffle=False,   # streaming order matters
            pin_memory=True,
        )

    else:
        raise ValueError("Dataset is not from the chosen list")

    return test_loader, dataset.classnames, dataset.template
