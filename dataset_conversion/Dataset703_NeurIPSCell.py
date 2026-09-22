import argparse
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile as tif
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from skimage import io, segmentation

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

SUPPORTED_IMAGE_SUFFIXES = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


def _discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    label_files = sorted(list(root.rglob("*_label.tiff")) + list(root.rglob("*_label.tif")))
    if not label_files:
        raise FileNotFoundError(
            f"No instance label files matching '*_label.tiff' or '*_label.tif' found under {root}"
        )

    pairs = []
    seen_case_ids = set()
    for label_file in label_files:
        suffix = "_label" + label_file.suffix
        case_id = label_file.name[: -len(suffix)]
        local_candidates = [
            p for p in label_file.parent.glob(case_id + ".*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        recursive_candidates = [
            p for p in root.rglob(case_id + ".*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        candidates = sorted({*local_candidates, *recursive_candidates})
        if len(candidates) == 1:
            image_file = candidates[0]
        elif len(candidates) == 0:
            raise FileNotFoundError(
                f"Could not find RGB image for {label_file}. "
                f"Expected a file named {case_id} with one of the extensions: "
                f"{sorted(SUPPORTED_IMAGE_SUFFIXES)}"
            )
        else:
            raise RuntimeError(f"Found multiple image candidates for {label_file}: {candidates}")

        if case_id in seen_case_ids:
            raise RuntimeError(f"Duplicate case id detected: {case_id}")
        seen_case_ids.add(case_id)
        pairs.append((image_file, label_file))
    return pairs


def _instance_to_three_class(label_file: Path) -> np.ndarray:
    inst = tif.imread(label_file)
    inst = np.asarray(inst).squeeze()
    if inst.ndim != 2:
        raise ValueError(f"Expected a 2D instance map in {label_file}, got shape {inst.shape}")
    inst = inst.astype(np.int32, copy=False)

    foreground = inst > 0
    boundary = segmentation.find_boundaries(inst, connectivity=1, mode="inner") & foreground

    seg = np.zeros(inst.shape, dtype=np.uint8)
    seg[foreground] = 1
    seg[boundary] = 2
    return seg


def _prepare_image_for_png(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image

    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32)
    elif np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise TypeError(f"Unsupported image dtype for PNG conversion: {image.dtype}")

    img_min = float(image.min())
    img_max = float(image.max())

    if img_min >= 0.0 and img_max <= 1.0:
        image = image * 255.0
    elif not (img_min >= 0.0 and img_max <= 255.0):
        image = image - img_min
        denom = float(image.max())
        if denom > 0:
            image = image / denom
        image = image * 255.0

    image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return image


def _ensure_three_channels(image: np.ndarray, image_file: Path) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image array for {image_file}, got shape {image.shape}")

    if image.shape[-1] == 3:
        return image
    if image.shape[-1] == 4:
        return image[..., :3]
    if image.shape[-1] == 1:
        return np.repeat(image, 3, axis=2)

    raise ValueError(f"Unsupported channel count for {image_file}: shape {image.shape}")


def convert_training_labeled(
    zip_path: Path,
    dataset_name: str = "Dataset703_NeurIPSCell",
    extract_dir: Path = None,
    overwrite: bool = False,
):
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")

    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    if extract_dir.exists() and any(extract_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Extract dir {extract_dir} already exists and is not empty. "
            "Use --overwrite if you want to recreate it."
        )

    if overwrite and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    pairs = _discover_pairs(extract_dir)

    dataset_root = Path(nnUNet_raw) / dataset_name
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    labels_inst = dataset_root / "labelsTr_instance"

    if overwrite:
        shutil.rmtree(images_tr, ignore_errors=True)
        shutil.rmtree(labels_tr, ignore_errors=True)
        shutil.rmtree(labels_inst, ignore_errors=True)

    maybe_mkdir_p(str(images_tr))
    maybe_mkdir_p(str(labels_tr))
    maybe_mkdir_p(str(labels_inst))

    for image_file, label_file in pairs:
        case_id = image_file.stem
        out_image = images_tr / f"{case_id}_0000.png"
        out_label = labels_tr / f"{case_id}.png"
        out_inst = labels_inst / f"{case_id}_label.tiff"

        image = _prepare_image_for_png(io.imread(str(image_file)))
        image = _ensure_three_channels(image, image_file)
        io.imsave(str(out_image), image, check_contrast=False)
        io.imsave(str(out_label), _instance_to_three_class(label_file), check_contrast=False)
        shutil.copy2(label_file, out_inst)

    generate_dataset_json(
        str(dataset_root),
        {0: "R", 1: "G", 2: "B"},
        {"background": 0, "interior": 1, "boundary": 2},
        num_training_cases=len(pairs),
        file_ending=".png",
        dataset_name=dataset_name,
        description=(
            "Converted from NeurIPS 2022 Cell Segmentation Challenge Training-labeled.zip. "
            "labelsTr are 3-class semantic masks derived from instance labels; "
            "original instance labels are kept in labelsTr_instance for F1 evaluation."
        ),
    )

    print(f"Extracted archive to: {extract_dir}")
    print(f"Prepared nnU-Net dataset at: {dataset_root}")
    print(f"Training cases: {len(pairs)}")
    print(f"Original instance labels copied to: {labels_inst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert NeurIPS 2022 CellSeg Training-labeled.zip into nnU-Net Dataset703 format."
    )
    parser.add_argument(
        "--zip_path",
        required=True,
        type=Path,
        help="Path to Training-labeled.zip",
    )
    parser.add_argument(
        "--extract_dir",
        type=Path,
        default=None,
        help="Where to extract the archive. Default: zip_path without .zip",
    )
    parser.add_argument(
        "--dataset_name",
        default="Dataset703_NeurIPSCell",
        type=str,
        help="Target nnU-Net dataset name",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files and converted dataset folders",
    )
    args = parser.parse_args()
    convert_training_labeled(args.zip_path, args.dataset_name, args.extract_dir, args.overwrite)
