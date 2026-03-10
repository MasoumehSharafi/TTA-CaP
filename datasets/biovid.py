import os
from pathlib import Path
from .utils import Datum, DatasetBase

# Two-class setting
template = ["a person with an expression of {}."]  # you can change later

CLASSNAMES = ["neutral", "pain"]
CLASS2IDX = {"neutral": 0, "pain": 1}


class BioVid(DatasetBase):
    """
    Root expected:
      <root>/BioVid_Video/
          train/ ...
          validation/ ...
          1/
            neutral/ <video_folders>/ <frames>
            pain/    <video_folders>/ <frames>
          2/
          ...
          10/
    For TTA/test, we ONLY use subject folders (digits), ignore train/validation.
    """

    dataset_dir = "BioVid_Video"

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        test = self._read_subject_folders(self.dataset_dir)
        super().__init__(test=test)

    @staticmethod
    def _is_subject_folder(name: str) -> bool:
        # subject folders are "1", "2", ..., "10" (digits)
        return name.isdigit()

    def _read_subject_folders(self, dataset_dir: str):
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"BioVid_Video not found at: {dataset_dir}")

        items = []
        subject_dirs = sorted([p for p in dataset_dir.iterdir()
                               if p.is_dir() and self._is_subject_folder(p.name)])

        for subj_dir in subject_dirs:
            subj_id = int(subj_dir.name)  # store as domain if you want

            for cname in CLASSNAMES:
                cls_dir = subj_dir / cname
                if not cls_dir.exists():
                    continue

                # recursively collect frames under neutral/ or pain/
                for img_path in sorted(cls_dir.rglob("*.jpg")) + sorted(cls_dir.rglob("*.png")) + sorted(cls_dir.rglob("*.jpeg")):
                    items.append(
                        Datum(
                            impath=str(img_path),
                            label=CLASS2IDX[cname],
                            domain=subj_id,
                            classname=cname
                        )
                    )

        if len(items) == 0:
            raise RuntimeError(
                f"No images found under subject folders in {dataset_dir}. "
                f"Expected e.g. {dataset_dir}/1/neutral/**.jpg"
            )

        return items
