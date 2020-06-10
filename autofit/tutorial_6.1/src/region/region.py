import numpy as np


class Region(np.ndarray):

    def __new__(
        cls, array_1d, name=""
    ):

        array_1d = array_1d.astype("bool")
        obj = array_1d.view(cls)

        # if molecule in [
        #     "CO",
        #     "C+"
        # ]:
        #     obj.molecule = molecule
        # else:
        #     raise ValueError("...")

        obj.name = name

        return obj

    @property
    def n_channels(self):
        return self.shape[0]

    @classmethod
    def manual(
        cls, array_1d
    ):

        return Region(
            array_1d=array_1d,
        )


    @classmethod
    def manual(
        cls, array_1d
    ):

        return Region(
            array_1d=array_1d,
        )


    @classmethod
    def from_limits(
        cls, n, n_min, n_max
    ):

        array_1d = np.zeros(
            shape=(int(n), ),
            dtype=int
        )

        if n_min > 0 and n_min < n_max and n_max < n:
            array_1d[n_min:n_max] = 1
        else:
            raise ValueError("...")

        return cls.manual(
            array_1d=array_1d,
        )



# def region(n, n_min, n_max, invert=False):
#
#     mask = np.zeros(
#         shape=int(n), dtype=int
#     )
#
#     if n_min > 0 and n_min < n_max and n_max < n:
#         mask[n_min:n_max] = 1
#     else:
#         raise ValueError("...")
#
#     return mask.astype(bool)

if __name__ == "__main__":

    region = Region.from_limits(n=32, n_min=8, n_max=24)
    print(region)
