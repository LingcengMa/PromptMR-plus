import contextlib
import os
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pathlib import Path
import h5py

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]
        
    def _get_ti_adj_idx_list(self,ti, num_t_in_volume):
        '''
        get the circular adjacent indices of the temporal axis for the given ti.
        '''
        start_lim, end_lim = -(num_t_in_volume//2), (num_t_in_volume//2+1)
        start, end = max(self.start_adj,start_lim), min(self.end_adj,end_lim)
        # Generate initial list of indices
        ti_idx_list = [(i + ti) % num_t_in_volume for i in range(start, end)]
        # duplicate padding if necessary
        replication_prefix = max(start_lim-self.start_adj,0) * ti_idx_list[0:1]
        replication_suffix = max(self.end_adj-end_lim,0) * ti_idx_list[-1:]

        return replication_prefix + ti_idx_list + replication_suffix
    
class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape. FastMRI multi-coil knee dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching. FastMRI multi-coil brain dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask

class FixedLowRandomMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines and a fixed number of low-frequency lines. CMRxRecon dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_low_frequencies, acceleration = self.choose_acceleration()
        num_low_frequencies = int(num_low_frequencies)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies

    def sample_kt_mask(self, shape, offset, num_adj_slices, slice_idx, num_t,num_slc, rng):
        if not hasattr(self, 'start_adj'):
            self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1

        num_cols = shape[-2]
        num_low_frequencies = rng.choice(self.center_fractions)
        acceleration = rng.choice(self.accelerations)

        mask = []
        for _ in range(num_t): #num_adj_slices
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                rng.uniform(size=num_cols) < 1/acceleration, ##* use the rng from cmrxrecon24maskfunc
                shape,
            )
            mask.append(torch.max(center_mask, acceleration_mask))


        mask = torch.cat(mask, dim=0)

        ti = slice_idx//num_slc
        select_list = self._get_ti_adj_idx_list(ti,num_t)
        mask = mask[select_list]
    
        return mask, num_low_frequencies

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        prob = 1/acceleration

        return self.rng.uniform(size=num_cols) < prob
        

class FixedLowEquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines and a fixed number of low-frequency lines. CMRxRecon dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_low_frequencies, acceleration = self.choose_acceleration()
        num_low_frequencies = int(num_low_frequencies)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies

    def sample_uniform_mask(self, shape, offset, rng):

        num_cols = shape[-2]
        num_low_frequencies = rng.choice(self.center_fractions)

        acceleration = rng.choice(self.accelerations)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )
        mask = torch.max(center_mask, acceleration_mask)
        return mask, num_low_frequencies

    def sample_kt_mask(self, shape, offset, num_adj_slices, slice_idx, num_t,num_slc, rng, seed):
        ##* important: need to use the rng from cmrxrecon24maskfunc; so validation is reproduceable; 

        if not hasattr(self, 'start_adj'):
            self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1

        num_cols = shape[-2]
        num_low_frequencies = rng.choice(self.center_fractions)
        acceleration = rng.choice(self.accelerations)

        if offset is None:
            offset=0
        num_low_frequencies = int(num_low_frequencies)
        if seed is None: ##* training
            ti = rng.randint(num_t)
        else: ##* validation
            ti = slice_idx//num_slc
        select_list = self._get_ti_adj_idx_list(ti,num_t)
        mask = []
        for _offset in select_list:
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                self.calculate_acceleration_mask(
                    num_cols, acceleration, _offset%acceleration, num_low_frequencies
                ),
                shape,
            )
            mask.append(torch.max(center_mask, acceleration_mask))
        mask = torch.cat(mask, dim=0)

        return mask, num_low_frequencies

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset=0

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask

class PoissonDiscMaskFunc(MaskFunc):
    """
    Sample data with Poisson-Disc sampling which is used in Calgary-Campinas dataset.

    """
    def __init__(
        self,
        center_radii: Sequence[int],
        accelerations: Sequence[int],
        mask_path: str,
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_radii) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )
        self.masks_path = Path(mask_path)
        self.center_radii = center_radii
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

        if not all(_ in [5, 10] for _ in accelerations):
            raise ValueError("CalgaryCampinas only provide 5x and 10x acceleration masks.")

        self.masks = {}
        self.centered_masks = {}
        for acceleration in accelerations:
            self.masks[acceleration] = self._load_masks(acceleration)

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_radii), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_radii))
            return self.center_radii[choice], self.accelerations[choice]

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            mask, radius_low_frequencies = self.sample_mask(
                shape, offset
            )

        return mask, radius_low_frequencies

    def sample_mask(self,shape,offset=None):

        shape_hw = shape[-3:-1] # (h,w)
        center_radius, acceleration = self.choose_acceleration()
        masks =  self.masks[acceleration]

        mask, num_masks = masks[shape_hw]
        # Randomly pick one example
        choice = self.rng.randint(0, num_masks)

        return torch.from_numpy(mask[choice][np.newaxis, ..., np.newaxis]), center_radius
    
    def circular_centered_mask(self,shape, radius):
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = ((dist_from_center <= radius) * np.ones(shape)).astype(bool)
        return mask[np.newaxis, ..., np.newaxis]
    
    def _load_masks(self, acceleration):
        paths = [
            f"R{acceleration}_218x170.npy",
            f"R{acceleration}_218x174.npy",
            f"R{acceleration}_218x180.npy",
        ]

        output = {}

        for path in paths:
            shape = path.split("_")[-1][:-4].split("x")
            shape = (int(shape[0]), int(shape[1]))
            mask_array = np.load(self.masks_path / path)
            output[shape] = mask_array.astype(np.float32), mask_array.shape[0]

        return output

class CmrxRecon24MaskFunc(MaskFunc):
    """
    Sample data 

    """
    def __init__(
        self,
        num_low_frequencies: Sequence[int],
        num_adj_slices: int,
        mask_path: Optional[str] = None,
        allowed_mask_types: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """

        self.num_low_frequencies = num_low_frequencies
        self.seed = seed
        self.uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [4, 8, 10], allow_any_combination=True, seed=seed)
        self.kt_uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [4, 8, 12, 16, 20, 24], allow_any_combination=True, seed=seed)
        self.kt_random_mask = FixedLowRandomMaskFunc(num_low_frequencies, [4, 8, 12, 16, 20, 24], allow_any_combination=True, seed=seed)
        self.radial_mask_bank = self._load_masks(mask_path, required=False)

        # mask_dict is set according to cmrxrecon24 challenge settings
        full_mask_dict = {
            "uniform": [4, 8, 10],
            "kt_uniform": [4, 8, 12, 16, 20, 24],
            "kt_random": [4, 8, 12, 16, 20, 24],
            "kt_gaussian": [4, 8, 12, 16, 20, 24],
            "kt_radial": [4, 8, 12, 16, 20, 24],
        }
        if allowed_mask_types is None:
            self.mask_dict = full_mask_dict
        else:
            invalid_types = sorted(set(allowed_mask_types) - set(full_mask_dict.keys()))
            if invalid_types:
                raise ValueError(
                    f"`allowed_mask_types` contains unsupported mask types: {invalid_types}. "
                    f"Supported mask types: {sorted(full_mask_dict.keys())}."
                )
            self.mask_dict = {mask_type: full_mask_dict[mask_type] for mask_type in allowed_mask_types}
            if not self.mask_dict:
                raise ValueError("`allowed_mask_types` cannot be empty.")
        self.masks_pool = list(self.mask_dict.keys())

        self._warned_generated_radial = False

        self.rng = np.random.RandomState(seed)

        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(num_adj_slices // 2), num_adj_slices // 2 + 1

    def choose_mask(self):
        '''
        choose from FixedLowEquiSpacedMaskFunc, FixedLowRandomMaskFunc and radial
        '''
        mask_type = self.rng.choice(self.masks_pool)
        return mask_type

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        slice_idx: Optional[int] = None,
        num_t: Optional[int] = None,
        num_slc: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")
        self.seed = seed
        with temp_seed(self.rng, seed):
            mask_type = self.choose_mask()
            mask, num_low_frequencies = self.sample_mask(mask_type, shape, offset, slice_idx, num_t, num_slc)

        return mask, num_low_frequencies, mask_type

    def sample_mask(self, mask_type, shape, offset=None, slice_idx=None, num_t=None, num_slc=None):
        if mask_type == "uniform":
            mask, num_low_frequencies = self.uniform_mask.sample_uniform_mask(shape, offset, self.rng)
        elif mask_type == "kt_uniform":
            mask, num_low_frequencies = self.kt_uniform_mask.sample_kt_mask(
                shape, offset, self.num_adj_slices, slice_idx, num_t, num_slc, self.rng, self.seed
            )
        elif mask_type == "kt_random":
            mask, num_low_frequencies = self.kt_random_mask.sample_kt_mask(
                shape, offset, self.num_adj_slices, slice_idx, num_t, num_slc, self.rng
            )
        elif mask_type == "kt_gaussian":
            if num_t is None or num_slc is None:
                raise ValueError(
                    "`num_t` and `num_slc` must be provided for `kt_gaussian` mask sampling."
                )
            h, w = shape[-3:-1]
            acc = self.rng.choice(self.mask_dict[mask_type])
            num_low_frequencies = self.rng.choice(self.num_low_frequencies)
            mask_ = self._generate_gaussian_vd_masks(num_t=num_t, h=h, w=w, acc=acc)

            if self.seed is None:  # training
                ti = self.rng.randint(num_t)
            else:  # validation
                if slice_idx is None:
                    raise ValueError("`slice_idx` must be provided when seed is set for validation sampling.")
                ti = slice_idx // num_slc
            select_list = self._get_ti_adj_idx_list(ti, num_t)
            mask = mask_[select_list][..., None]
        elif mask_type == "kt_radial":
            # TODO: wrap this path in a dedicated MaskFunc like the other mask types.
            h, w = shape[-3:-1]
            acc = self.rng.choice(self.mask_dict[mask_type])
            num_low_frequencies = self.rng.choice(self.num_low_frequencies)
            radial_key = f"acc{acc}_{w}x{h}"
            if radial_key in self.radial_mask_bank:
                mask_ = self.radial_mask_bank[radial_key][:num_t]
            else:
                if not self._warned_generated_radial:
                    warnings.warn(
                        "Radial mask bank unavailable for requested shape/acceleration. "
                        "Falling back to on-the-fly pseudo-radial mask generation.",
                        RuntimeWarning,
                    )
                    self._warned_generated_radial = True
                mask_ = self._generate_pseudo_radial_masks(num_t, h, w, acc)

            if num_t is None or num_slc is None:
                raise ValueError(
                    "`num_t` and `num_slc` must be provided for `kt_radial` mask sampling."
                )

            if self.seed is None: ##* training
                ti = self.rng.randint(num_t)
            else: ##* validation
                ##* slice_idx is of range(num_t*num_slc)
                if slice_idx is None:
                    raise ValueError("`slice_idx` must be provided when seed is set for validation sampling.")
                ti = slice_idx // num_slc
            select_list = self._get_ti_adj_idx_list(ti, num_t)

            mask = mask_[select_list]
            mask = mask[..., None]  # torch.Size([5, 448, 204, 1])

        else:
            raise ValueError(f"{mask_type} not supported")

        return mask.float(), num_low_frequencies

    def _generate_gaussian_vd_masks(
        self,
        num_t: int,
        h: int,
        w: int,
        acc: int,
        center_keep_radius: int = 4,
    ) -> torch.Tensor:
        """
        Procedurally generate variable-density Gaussian kt masks.
        Output shape: [num_t, h, w].
        """
        if num_t is None:
            raise ValueError("`num_t` is required for kt_gaussian mask generation.")

        total_points = max(1, (h * w) // max(acc, 1))
        sigma_x = w / 5.0
        sigma_y = h / 5.0

        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0

        prob = np.exp(
            -((xx - cx) ** 2) / (2 * sigma_x**2)
            -((yy - cy) ** 2) / (2 * sigma_y**2)
        )
        prob = prob / prob.sum()
        flat_prob = prob.ravel()
        n_total = flat_prob.size

        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        center_mask = dist2 <= center_keep_radius**2

        masks = np.zeros((num_t, h, w), dtype=np.float32)
        for t in range(num_t):
            idx = self.rng.choice(n_total, size=min(total_points, n_total), replace=False, p=flat_prob)
            frame = np.zeros(n_total, dtype=np.float32)
            frame[idx] = 1.0
            frame = frame.reshape(h, w)
            frame[center_mask] = 1.0
            masks[t] = frame

        return torch.from_numpy(masks)

    def _generate_pseudo_radial_masks(self, num_t: int, h: int, w: int, acc: int) -> torch.Tensor:
        """
        Procedurally generate pseudo-radial masks when precomputed masks are unavailable.
        Output shape: [num_t, h, w].
        """
        if num_t is None:
            raise ValueError("`num_t` is required for kt_radial mask generation.")

        yy, xx = np.mgrid[0:h, 0:w]
        yy = yy - (h - 1) / 2.0
        xx = xx - (w - 1) / 2.0

        num_spokes = max(1, int(round(min(h, w) / max(acc, 1))))
        line_width = 0.6

        masks = np.zeros((num_t, h, w), dtype=np.float32)
        for t in range(num_t):
            # Deterministic temporal angle shift keeps a dynamic kt-radial pattern.
            phase = (t / max(num_t, 1)) * np.pi
            angles = np.linspace(0.0, np.pi, num=num_spokes, endpoint=False) + phase

            frame_mask = np.zeros((h, w), dtype=bool)
            for theta in angles:
                distance_to_line = np.abs(xx * np.cos(theta) + yy * np.sin(theta))
                frame_mask |= distance_to_line <= line_width

            masks[t] = frame_mask.astype(np.float32)

        return torch.from_numpy(masks)

    def _load_masks(self, mask_path, required: bool = False):
        """Load CMRxRecon24 pseudo-radial masks from an h5 file."""
        candidate_paths = []
        if mask_path:
            candidate_paths.append(Path(mask_path).expanduser())

        env_mask_path = os.environ.get("PROMPTMR_CMRX_MASK_PATH")
        if env_mask_path:
            candidate_paths.append(Path(env_mask_path).expanduser())

        repo_root = Path(__file__).resolve().parents[1]
        candidate_paths.extend(
            [
                repo_root / "mask_radial.h5",
                repo_root / "mask_files_required_for_training" / "mask_radial.h5",
            ]
        )

        valid_paths = [path for path in candidate_paths if path.is_file()]
        if not valid_paths:
            searched_paths = "\n".join(f"  - {path}" for path in candidate_paths) or "  - (none)"
            msg = (
                "CMRxRecon radial mask file not found.\n"
                "Searched the following paths:\n"
                f"{searched_paths}\n"
                "Provide `mask_path`, set `PROMPTMR_CMRX_MASK_PATH`, or place `mask_radial.h5` "
                "at the project root. Download link is documented in DATASET.md."
            )
            if required:
                raise FileNotFoundError(msg)
            warnings.warn(
                f"{msg}\nWill fall back to on-the-fly pseudo-radial mask generation when needed.",
                RuntimeWarning,
            )
            return {}
        mask_path = valid_paths[0]

        radial_mask_bank = {}
        try:
            with h5py.File(mask_path, "r") as hf:
                keys = list(hf.keys())
                for key_ in keys:
                    radial_mask_bank[key_] = torch.from_numpy(hf[key_][()].transpose(0, 2, 1))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to open CMRxRecon radial mask file '{mask_path}': {exc}."
            ) from exc
        return radial_mask_bank

class CmrxRecon24TestValMaskFunc(CmrxRecon24MaskFunc):
    """
    Sample data 
    """
    def __init__(
        self,
        num_low_frequencies: Sequence[int],
        num_adj_slices: int,
        mask_path: Optional[str] = None,
        allowed_mask_types: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        test_mask_type: str = 'uniform',
        test_acc: int = 10
    ):

        self.num_low_frequencies = num_low_frequencies
        self.seed = seed
        self.uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed)
        self.kt_uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed)
        self.kt_random_mask = FixedLowRandomMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed)
        self.radial_mask_bank = self._load_masks(mask_path, required=False)

        # mask_dict is set according to test config
        self.mask_dict = {test_mask_type: [test_acc]}
        if allowed_mask_types is not None:
            invalid_types = sorted(set(allowed_mask_types) - set(self.mask_dict.keys()))
            if invalid_types:
                raise ValueError(
                    f"`allowed_mask_types` contains unsupported mask types for test/val: {invalid_types}. "
                    f"Supported mask types in this configuration: {sorted(self.mask_dict.keys())}."
                )
        self.masks_pool = list(self.mask_dict.keys())

        self._warned_generated_radial = False

        self.rng = np.random.RandomState(seed)

        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1
