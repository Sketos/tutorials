import autolens as al


class Grid3D:
    def __init__(self, grid_2d, n_channels):

        self.grid_2d = grid_2d
        self.n_channels = n_channels

    @property
    def shape_2d(self):
        return self.grid_2d.shape_2d

    @property
    def shape_3d(self):
        return (self.n_channels, ) + self.grid_2d.shape_2d

    @property
    def pixel_scale(self):
        return self.grid_2d.pixel_scale

    @property
    def pixel_scales(self):
        return self.grid_2d.pixel_scales

    @property
    def sub_size(self):
        return self.grid_2d.sub_size

    @property
    def extent(self):
        return [
            self.grid_2d.extent[0] - self.pixel_scale,
            self.grid_2d.extent[1] + self.pixel_scale,
            self.grid_2d.extent[2] - self.pixel_scale,
            self.grid_2d.extent[3] + self.pixel_scale
        ]
