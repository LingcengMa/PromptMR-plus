from pathlib import Path
from data.mri_data import CmrxReconSliceDataset

train_root = Path("/radraid/lingcengma/data/CMRxRecon2026/TaskR1R2/TrainSet/Aorta/Center007/GE_30T_Architect/P001")

print("Entries in train_root:")
for p in sorted(train_root.iterdir()):
    print(" ", p, "| is_file:", p.is_file(), "| suffix:", p.suffix)

ds = CmrxReconSliceDataset(
    root=train_root,
    challenge="multicoil",
    transform=None,
    num_adj_slices=5,
)

print("len(ds) =", len(ds))

sample = ds[0]
kspace, mask, target, attrs, fname, data_slice, num_t = sample

print("fname:", fname)
print("data_slice:", data_slice)
print("num_t:", num_t)
print("kspace shape:", kspace.shape)
print("target shape:", None if target is None else target.shape)
print("attrs['shape']:", attrs["shape"])