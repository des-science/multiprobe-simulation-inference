# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created January 2024
Author: Arne Thomsen

Utils to load MCMC chains to be plotted.
"""

import torch
from msfm.utils import files, parameters, parameters

# class CustomUniformPrior:
#     """User defined numpy uniform prior.

#     Custom prior with user-defined valid .sample and .log_prob methods.
#     """

#     def __init__(self, lower: Tensor, upper: Tensor, return_numpy: bool = False):
#         self.lower = lower
#         self.upper = upper
#         self.dist = BoxUniform(lower, upper)
#         self.return_numpy = return_numpy

#     def sample(self, sample_shape=torch.Size([])):
#         samples = self.dist.sample(sample_shape)
#         return samples.numpy() if self.return_numpy else samples

#     def log_prob(self, values):
#         if self.return_numpy:
#             values = torch.as_tensor(values)
#         log_probs = self.dist.log_prob(values)
#         return log_probs.numpy() if self.return_numpy else log_probs


def torch_in_grid_prior(cosmos, conf=None, params=None, device="cpu"):
    """Determines whether the elements of the given array of cosmological parameters are contained within the analysis
    prior. This is needed to build a vectorized log posterior.

    Args:
        cosmos (np.ndarray): A 2D array of cosmological parameters with shape (n_cosmos, n_params), where n_params
            has to be in the right ordering (as defined in the config) and n_theta corresponds to n_cosmos.
        conf (str, dict, optional): Config to use, can be either a string to the config.yaml file, the dictionary
            obtained by reading such a file or None, where the default config within the repo is used. Defaults to
            None.
        params (list, optional): List of strings containing "Om", "s8", "Ob", "H0", "ns" and "w0" in the same order as
            within the cosmos array.

    Raises:
        ValueError: If an incompatible type is passed to the conf argument

    Returns:
        in_prior: A 1D boolean array of the shape (n_cosmos,) that specifies whether the values in params are
        contained within the prior.
    """
    conf = files.load_config(conf)
    params = parameters.get_parameters(params, conf)

    # make the params 2d
    cosmos = torch.atleast_2d(cosmos)

    prior_intervals = torch.tensor(parameters.get_prior_intervals(params), dtype=torch.float32, device=device)

    # check if we are in the prior intervals
    in_prior = torch.all(torch.logical_and(prior_intervals[:, 0] <= cosmos, cosmos <= prior_intervals[:, 1]), axis=1)

    # simplex in the Om - s8 plane
    try:
        i_Om = params.index("Om")
        i_s8 = params.index("s8")
    except ValueError:
        pass
    else:
        Om_s8_border_points = torch.tensor(
            conf["analysis"]["grid"]["priors"]["Om_s8_border_points"], dtype=torch.float32, device=device
        )

        in_hull = is_inside_hull(Om_s8_border_points, cosmos[:, [i_Om, i_s8]])

        # what is False will stay false irrespective of the rhs
        in_prior_clone = in_prior.clone()
        in_prior_clone[in_prior] = in_hull[in_prior]
        in_prior = in_prior_clone

    # w0 threshold
    try:
        i_Om = params.index("Om")
        i_w0 = params.index("w0")
    except ValueError:
        pass
    else:
        # check if we are above the w0 threshold (same as get_min_w0 with margin = 0.01)
        in_prior[in_prior] = 1.0 / (cosmos[in_prior, i_Om] - 1.0) + 0.01 <= cosmos[in_prior, i_w0]

    return in_prior

def is_inside_hull(hull_points, query_points):
    hull_points = hull_points.unsqueeze(0)  # shape: (1, n_hull_points, 2)
    query_points = query_points.unsqueeze(1)  # shape: (n_query_points, 1, 2)

    # Compute vectors from query points to hull points
    vectors_to_query_points = query_points - hull_points  # shape: (n_query_points, n_hull_points, 2)

    # Compute vectors from each hull point to the next hull point
    vectors_to_next_hull_points = torch.roll(hull_points, shifts=-1, dims=1) - hull_points  # shape: (1, n_hull_points, 2)

    # Compute cross product of vectors
    cross_products = vectors_to_query_points[..., 0] * vectors_to_next_hull_points[..., 1] - vectors_to_query_points[..., 1] * vectors_to_next_hull_points[..., 0]  # shape: (n_query_points, n_hull_points)

    # Check if all cross products for each query point have the same sign
    is_inside = (cross_products >= 0).all(dim=1) | (cross_products <= 0).all(dim=1)  # shape: (n_query_points,)

    return is_inside