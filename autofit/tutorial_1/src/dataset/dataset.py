from autoarray.structures import grids


class Dataset:
    def __init__(self, data, noise_map):

        self.data = data
        self.noise_map = noise_map


class MaskedDataset:
    def __init__(self, dataset, mask):

        self.dataset = dataset
        # TODO: Check that dataset has attribute data.

        # TODO: Check that mask and dataset.data have the same shape.
        self.mask = mask

        if mask.pixel_scales is not None:

            self.grid = grids.MaskedGrid.from_mask(mask=mask)

        else:

            raise ValueError("...")

        self.data = dataset.data

        self.noise_map = dataset.noise_map
