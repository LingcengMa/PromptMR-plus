import h5py
import numpy as np
from pathlib import Path

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
            print("Available keys:", list(f.keys()))
            raise KeyError(f"{var_name} not found in {mat_path}")
        arr = f[var_name][()]
    return arr

def convert_kdata_full_to_h5(
    mat_path,
    out_dir,
    mat_var="kdata_full",
    patient_id="P001",
    acquisition="flow",
):
    kdata_full = load_mat_v73_variable(mat_path, mat_var)

    # If MATLAB stored complex numbers in native complex dtype, this is enough.
    # If not, we will handle that below.
    if isinstance(kdata_full, np.ndarray) and kdata_full.dtype.names is not None:
        # compound dtype case: fields like ('real','imag')
        field_names = set(kdata_full.dtype.names)
        if {"real", "imag"}.issubset(field_names):
            kdata_full = kdata_full["real"] + 1j * kdata_full["imag"]

    if kdata_full.ndim != 6:
        raise ValueError(f"Expected 6D array [Nv,Nt,Nc,SPE,PE,FE], got {kdata_full.shape}")

    print("Loaded raw shape:", kdata_full.shape)

    Nv, Nt, Nc, SPE, PE, FE = kdata_full.shape
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for v in range(Nv):
        # raw one-encoding block: [Nt, Nc, SPE, PE, FE]
        k = kdata_full[v]

        # reorder to match CmrxReconSliceDataset expectations:
        # [Nt, SPE, Nc, FE, PE]
        k = np.transpose(k, (0, 2, 1, 4, 3))

        img_coil = ifft2c_np(k)
        img_rss = rss_np(img_coil, coil_axis=2)

        save_name = f"{patient_id}_{acquisition}_v{v}"
        out_path = out_dir / f"{save_name}.h5"

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
            f.attrs["patient_id"] = save_name
            f.attrs["acquisition"] = f"{acquisition}_v{v}"

        print(f"Saved {out_path}")
        print(f"  kspace shape: {k.shape}")
        print(f"  reconstruction_rss shape: {img_rss.shape}")

if __name__ == "__main__":
    convert_kdata_full_to_h5(
        mat_path="/radraid/amacintyre/data/CMRxRecon2026/TaskR1R2/TrainSet/Aorta/Center007/GE_30T_Architect/P001/kdata_full.mat",
        out_dir="/radraid/lingcengma/data/CMRxRecon2026/TaskR1R2/TrainSet/Aorta/Center007/GE_30T_Architect/P001/",
        mat_var="kdata_full",
        patient_id="P001",
        acquisition="flow",
    )