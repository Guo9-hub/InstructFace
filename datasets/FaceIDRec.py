# coding: utf-8
import os
import random
import numpy as np

# ============================================================
# Fix mxnet compatibility with newer numpy.
# ============================================================
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float

from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import mxnet as mx
except ImportError:
    mx = None


class InstructFaceIDRec(Dataset):
    """
    MS1MV2 / Glint360K / CASIA-WebFace RecordIO dataset.

    Reads:
        train.rec + train.idx

    Returns:
        image, identity_label, instruction, description

    This follows the stable InsightFace/TransFace++ RecordIO reading style:
    - use idx=0 header to get image index range;
    - decode images with mx.image.imdecode instead of PIL BytesIO.
    """

    def __init__(
        self,
        root,
        transform=None,
        instruct_json_path=None,
        ablation="full",
        rec_name="train.rec",
        idx_name="train.idx",
        use_general_description=True,
        max_retry=10,
        max_samples=None,
    ):
        super().__init__()

        if mx is None:
            raise ImportError(
                "mxnet is required for train.rec/train.idx. Please install mxnet first."
            )

        self.root = root
        self.transform = transform
        self.ablation = ablation
        self.use_general_description = use_general_description
        self.max_retry = max_retry

        self.rec_path = os.path.join(root, rec_name)
        self.idx_path = os.path.join(root, idx_name)

        if not os.path.exists(self.rec_path):
            raise FileNotFoundError(f"Cannot find rec file: {self.rec_path}")
        if not os.path.exists(self.idx_path):
            raise FileNotFoundError(f"Cannot find idx file: {self.idx_path}")

        self.imgrec = None
        self.num_classes_from_rec = None

        tmp_rec = mx.recordio.MXIndexedRecordIO(self.idx_path, self.rec_path, "r")
        try:
            header, _ = mx.recordio.unpack(tmp_rec.read_idx(0))
            if getattr(header, "flag", 0) > 0 and isinstance(header.label, (list, tuple, np.ndarray)):
                self.header0 = (int(header.label[0]), int(header.label[1]))
                self.num_classes_from_rec = int(header.label[1])
                self.indices = list(range(1, int(header.label[0])))
                print(f"[FaceIDRec] header label: {header.label}")
                print(f"[FaceIDRec] image index range: 1 ~ {int(header.label[0]) - 1}")
                print(f"[FaceIDRec] num_classes from rec: {self.num_classes_from_rec}")
            else:
                self.indices = [idx for idx in list(tmp_rec.keys) if idx != 0]
                print("[FaceIDRec][WARN] Standard header not found; fallback to tmp_rec.keys.")
        except Exception as e:
            self.indices = [idx for idx in list(tmp_rec.keys) if idx != 0]
            print(f"[FaceIDRec][WARN] Failed to read idx=0 header, fallback to keys. Error: {e}")
        finally:
            tmp_rec.close()

        if max_samples is not None and int(max_samples) > 0:
            self.indices = self.indices[:int(max_samples)]
            print(f"[FaceIDRec DEBUG] use subset samples: {len(self.indices)}")

        self.instructions = [
            "Face Recognition: identify the person in the face image.",
            "Face Recognition: learn identity-discriminative facial features.",
            "Face Recognition: recognize the identity represented by this face.",
            "Face Recognition: extract identity-related information from the face image.",
            "Face Recognition: distinguish this face from other identities."
        ]

        print(f"[FaceIDRec] root: {root}")
        print(f"[FaceIDRec] rec: {self.rec_path}")
        print(f"[FaceIDRec] idx: {self.idx_path}")
        print(f"[FaceIDRec] samples: {len(self.indices)}")

    def _init_rec(self):
        if self.imgrec is None:
            self.imgrec = mx.recordio.MXIndexedRecordIO(
                self.idx_path,
                self.rec_path,
                "r"
            )

    def __len__(self):
        return len(self.indices)

    def _read_one(self, i):
        self._init_rec()

        idx = self.indices[i]
        record = self.imgrec.read_idx(idx)
        if record is None:
            raise RuntimeError(f"read_idx({idx}) returned None.")

        header, img = mx.recordio.unpack(record)

        label = header.label
        if isinstance(label, (list, tuple, np.ndarray)):
            label = int(label[0])
        else:
            label = int(label)

        # TransFace++ / InsightFace style decoding.
        img_arr = mx.image.imdecode(img).asnumpy()
        image = Image.fromarray(img_arr).convert("RGB")

        return image, label

    def __getitem__(self, i):
        last_error = None

        for _ in range(self.max_retry):
            try:
                image, label = self._read_one(i)
                break
            except Exception as e:
                last_error = e
                i = random.randint(0, len(self.indices) - 1)
        else:
            raise RuntimeError(
                f"[FaceIDRec] Failed to read sample after {self.max_retry} retries. Last error: {last_error}"
            )

        if self.transform is not None:
            image = self.transform(image)

        instruction = random.choice(self.instructions)
        description = "A photo of a face."

        return image, torch.tensor(label, dtype=torch.long), instruction, description
