import numpy as np
import matplotlib.pyplot as plt


class DatasetFit:

    def __init__(self, masked_dataset, model_data, inversion=None):

        self.masked_dataset = masked_dataset

        if not (self.masked_dataset.data.shape == model_data.shape):
            raise ValueError(
                "The shape of the data and model_data arrays are not equal"
            )
        else:
            self.model_data = model_data

        self.inversion = inversion

    @property
    def mask(self):
        return self.masked_dataset.uv_mask

    @property
    def mask_real_and_imag_averaged(self):
        return self.masked_dataset.uv_mask_real_and_imag_averaged

    @property
    def data(self):
        return self.masked_dataset.data # NOTE: The data are now visibilities ...

    @property
    def noise_map(self):
        return self.masked_dataset.noise_map

    @property
    def noise_map_real_and_imag_averaged(self):
        return self.masked_dataset.noise_map_real_and_imag_averaged

    @property
    def residual_map(self):
        return residual_map_from_data_model_data_and_mask(
            data=self.data, model_data=self.model_data, mask=self.mask
        )

    @property
    def normalized_residual_map(self):
        return normalized_residual_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
        )

    @property
    def chi_squared_map(self):
        return chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
        )

    @property
    def signal_to_noise_map(self):
        signal_to_noise_map = np.divide(
            self.data,
            self.noise_map
        )
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def chi_squared(self):
        return chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=self.chi_squared_map, mask=self.mask
        )

    @property
    def noise_normalization(self):
        return noise_normalization_from_noise_map_and_mask(
            noise_map=self.noise_map_real_and_imag_averaged, mask=self.mask_real_and_imag_averaged
        )

    @property
    def likelihood(self):
        return likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )

    @property
    def likelihood_with_regularization(self):
        if self.inversion is not None:
            return likelihood_with_regularization_from_inversion_terms(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def evidence(self):
        if self.inversion is not None:
            return evidence_from_inversion_terms(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def figure_of_merit(self):
        if self.inversion is None:
            return self.likelihood
        else:
            return self.evidence


def residual_map_from_data_model_data_and_mask(data, mask, model_data):

    return np.subtract(
        data, model_data, out=np.zeros_like(data), where=np.asarray(mask) == 0
    )


def normalized_residual_map_from_residual_map_noise_map_and_mask(
    residual_map, noise_map, mask
):

    return np.divide(
        residual_map,
        noise_map,
        out=np.zeros_like(residual_map),
        where=np.asarray(mask) == 0,
    )


def chi_squared_map_from_residual_map_noise_map_and_mask(residual_map, noise_map, mask):

    return np.square(
        np.divide(
            residual_map,
            noise_map,
            out=np.zeros_like(residual_map),
            where=np.asarray(mask) == 0,
        )
    )

def chi_squared_from_chi_squared_map_and_mask(chi_squared_map, mask):

    return np.sum(chi_squared_map[np.asarray(mask) == 0])


def noise_normalization_from_noise_map_and_mask(noise_map, mask):

    return np.sum(np.log(2.0 * np.pi * noise_map[np.asarray(mask) == 0] ** 2.0))


def likelihood_from_chi_squared_and_noise_normalization(
    chi_squared, noise_normalization
):

    return -0.5 * (chi_squared + noise_normalization)


def likelihood_with_regularization_from_inversion_terms(
    chi_squared, regularization_term, noise_normalization
):

    return -0.5 * (chi_squared + regularization_term + noise_normalization)


def evidence_from_inversion_terms(
    chi_squared,
    regularization_term,
    log_curvature_regularization_term,
    log_regularization_term,
    noise_normalization,
):

    return -0.5 * (
        chi_squared
        + regularization_term
        + log_curvature_regularization_term
        - log_regularization_term
        + noise_normalization
    )
