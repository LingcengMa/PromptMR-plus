import h5py
import numpy as np
from pathlib import Path
import random


def ifft2c_np(kspace):
    x = np.fft.ifftshift(kspace, axes=(-2, -1))
    x = np.fft.ifft2(x, axes=(-2, -1), norm="ortho")
    x = np.fft.fftshift(x, axes=(-2, -1))
    return x


def rss_np(img, coil_axis=2):
    return np.sqrt(np.sum(np.abs(img) ** 2, axis=coil_axis))


def load_mat_v73_variable(mat_path, var_name):
    with h5py.File(mat_path, "r") as f:
        if var_name not in f:
            print(f"Available keys in {mat_path}: {list(f.keys())}")
            raise KeyError(f"{var_name} not found in {mat_path}")
        arr = f[var_name][()]
    return arr


def load_complex_mat_variable(mat_path, var_name):
    arr = load_mat_v73_variable(mat_path, var_name)

    # MATLAB compound complex storage case
    if isinstance(arr, np.ndarray) and arr.dtype.names is not None:
        field_names = set(arr.dtype.names)
        if {"real", "imag"}.issubset(field_names):
            arr = arr["real"] + 1j * arr["imag"]

    return arr


def convert_one_subject(
    subject_dir: Path,
    h5_dir: Path,
    acquisition: str = "flow",
    mat_var: str = "kdata_full",
):
    patient_id = subject_dir.name
    mat_path = subject_dir / "kdata_full.mat"

    if not mat_path.exists():
        print(f"[SKIP] Missing {mat_path}")
        return []

    kdata_full = load_complex_mat_variable(mat_path, mat_var)

    if kdata_full.ndim != 6:
        raise ValueError(
            f"{patient_id}: Expected 6D [Nv,Nt,Nc,SPE,PE,FE], got {kdata_full.shape}"
        )

    Nv, Nt, Nc, SPE, PE, FE = kdata_full.shape
    print(f"[INFO] {patient_id}: loaded shape {kdata_full.shape}")

    saved_files = []

    for v in range(Nv):
        # raw block: [Nt, Nc, SPE, PE, FE]
        k = kdata_full[v]

        # reorder to [Nt, SPE, Nc, FE, PE]
        k = np.transpose(k, (0, 2, 1, 4, 3))

        img_coil = ifft2c_np(k)
        img_rss = rss_np(img_coil, coil_axis=2)

        save_name = f"{patient_id}_{acquisition}_v{v}.h5"
        out_path = h5_dir / save_name

        with h5py.File(out_path, "w") as f:
            f.create_dataset("kspace", data=k.astype(np.complex64))
            f.create_dataset("reconstruction_rss", data=img_rss.astype(np.float32))

            f.attrs["shape"] = k.shape
            f.attrs["max"] = float(np.max(img_rss))
            f.attrs["norm"] = float(np.linalg.norm(img_rss))
            f.attrs["padding_left"] = 0
            f.attrs["padding_right"] = k.shape[-1]
            f.attrs["encoding_size"] = (k.shape[-2], k.shape[-1], 1)
            f.attrs["recon_size"] = (k.shape[-2], k.shape[-1], 1)
            f.attrs["patient_id"] = patient_id
            f.attrs["acquisition"] = f"{acquisition}_v{v}"

        saved_files.append(out_path)
        print(f"[SAVE] {out_path.name} | kspace {k.shape} | target {img_rss.shape}")

    return saved_files


def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(Path("..") / "h5_dataset" / src.name)


def build_promptmr_dataset(
    source_root: str,
    output_root: str,
    acquisition: str = "flow",
    val_fraction: float = 0.2,
    seed: int = 1234,
):
    source_root = Path(source_root)
    output_root = Path(output_root)

    h5_dir = output_root / "h5_dataset"
    train_dir = output_root / "train"
    val_dir = output_root / "val"

    h5_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([p for p in source_root.iterdir() if p.is_dir() and p.name.startswith("P")])
    print(f"[INFO] Found {len(subject_dirs)} subject folders")

    # split by subject, not by file
    subject_ids = [p.name for p in subject_dirs]
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n_val = max(1, round(len(subject_ids) * val_fraction))
    val_subjects = set(subject_ids[:n_val])
    train_subjects = set(subject_ids[n_val:])

    print(f"[INFO] Train subjects: {len(train_subjects)}")
    print(f"[INFO] Val subjects:   {len(val_subjects)}")

    all_saved = []

    for subject_dir in subject_dirs:
        saved_files = convert_one_subject(
            subject_dir=subject_dir,
            h5_dir=h5_dir,
            acquisition=acquisition,
            mat_var="kdata_full",
        )
        all_saved.extend(saved_files)

        for src in saved_files:
            patient_id = subject_dir.name
            if patient_id in val_subjects:
                dst = val_dir / src.name
            else:
                dst = train_dir / src.name
            safe_symlink(src, dst)

    print(f"\n[DONE] Total H5 files: {len(all_saved)}")
    print(f"[DONE] Dataset root: {output_root}")
    print(f"[DONE] h5_dataset:   {h5_dir}")
    print(f"[DONE] train:        {train_dir}")
    print(f"[DONE] val:          {val_dir}")

    # save split record
    split_txt = output_root / "split_summary.txt"
    with open(split_txt, "w") as f:
        f.write("Validation subjects:\n")
        for sid in sorted(val_subjects):
            f.write(f"{sid}\n")
        f.write("\nTraining subjects:\n")
        for sid in sorted(train_subjects):
            f.write(f"{sid}\n")
    print(f"[DONE] Wrote split summary to {split_txt}")


if __name__ == "__main__":
    build_promptmr_dataset(
        source_root="/radraid/amacintyre/data/CMRxRecon2026/TaskR1R2/TrainSet/Aorta/Center007/GE_30T_Architect",
        output_root="/radraid/lingcengma/data/CMRxRecon2026/TaskR1R2/TrainSet/Aorta/Center007/GE_30T_Architect/promptmr_dataset",
        acquisition="flow",
        val_fraction=0.2,
        seed=1234,
    )