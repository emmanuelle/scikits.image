import numpy as np


def _compute_gradient_weights(image, beta=1):
    """
    Compute weights on vertical and horizontal edges between neighboring
    pixels, using a Gaussian transformation of gradients magnitudes.

    Parameters
    ----------

    image : np.ndarray
        Image on which to compute the gradient

    beta : float
        Scaling parameter of gradients. The larger beta, the smaller the
        weight for edges with a nonzero gradient.
    """
    beta /= np.var(image)  # normalize beta
    weight_up = np.empty_like(image)
    weight_up[1:] = image[1:] - image[:-1]
    weight_up = np.exp(- beta * weight_up ** 2)
    weight_up[0] = 0
    weight_left = np.empty_like(image)
    weight_left[:, 1:] = image[:, 1:] - image[:, :-1]
    weight_left = np.exp(- beta * weight_left ** 2)
    weight_left[:, 0] = 0
    return weight_up, weight_left


def _number_neighbors(shape, mask=None):
    """
    Compute the connectivity of pixels in an image

    Parameters
    ----------

    shape : tuple
        shape of image in which to compute the connectivity

    mask : np.ndarray, default None
        boolean mask of pixels used for the graph of pixels

    """
    if mask is None:
        mask = np.ones(shape, dtype=np.uint8)
    neighbors = np.zeros_like(mask)
    neighbors[1:] += mask[:-1]
    neighbors[:-1] += mask[1:]
    neighbors[:, 1:] += mask[:, :-1]
    neighbors[:, :-1] += mask[:, 1:]
    return neighbors.astype(np.uint8)


def random_walker_local(image, markers, beta=1, n_iter=100):
    """
    Random walker algorithm for segmentation from markers.

    Parameters
    ----------

    image : array-like
        Image to be segmented

    markers : array-like, integer type
        Array of seed markers labeled with different positive integers
        for different phases. Zero-labeled pixels are unlabeled pixels.

    beta : float
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).

    n_iter : int
        Maximum number of iterations.


    Returns
    -------

    segmentation_result : np.ndarray
        array of labels assigned to pixels by the segmentation

    Notes
    -----

    For each possible label ``l``, the random walker algorithm computes the
    probability ``x`` that seeds labeled by ``l`` diffuse first to the
    unlabeled pixels. This is done by solving iteratively the fixed-point
    equation

    x_i = 1 / #{neighbors of i} sum_j w_{ij} x_j

    where w_{ij} is the weight on the edge i-j, j are the neighbors of i,
    and x_i is the probability to be computed.

    The iterations are stopped when, for all pixels, the label corresponding
    to the maximal probability does not change any more.

    In this implementation, only two labels are possible but it would be very
    easy to extend it to more labels.
    """
    image = image.astype(np.float)
    weight_up, weight_left = _compute_gradient_weights(image, beta)
    neighbors = _number_neighbors(image.shape)
    inds_bg = np.nonzero(markers == 1)
    inds_fg = np.nonzero(markers == 2)
    proba1 = 0.5 * np.ones_like(image)
    proba1[inds_bg] = 1
    proba1[inds_fg] = 0
    proba2 = 0.5 * np.ones_like(image)
    proba2[inds_bg] = 0
    proba2[inds_fg] = 1
    n_inner = 20
    n_step = n_iter / n_inner
    result = np.zeros_like(image)
    for i in range(n_step):
        print i
        for j in range(n_inner):
            new = np.zeros_like(proba1)
            new[1:] += proba1[:-1] * weight_up[1:]
            new[:-1] += proba1[1:] * weight_up[1:]
            new[:, 1:] += proba1[:, :-1] * weight_left[:, 1:]
            new[:, :-1] += proba1[:, 1:] * weight_left[:, 1:]
            new /= neighbors
            proba1 = new
            proba1[inds_bg] = 1
            proba1[inds_fg] = 0
        for j in range(n_inner):
            new = np.zeros_like(proba2)
            new[1:] += proba2[:-1] * weight_up[1:]
            new[:-1] += proba2[1:] * weight_up[1:]
            new[:, 1:] += proba2[:, :-1] * weight_left[:, 1:]
            new[:, :-1] += proba2[:, 1:] * weight_left[:, 1:]
            new /= neighbors
            proba2 = new
            proba2[inds_bg] = 0
            proba2[inds_fg] = 1
        segmentation_result = np.argmax([proba1, proba2], axis=0)
        if np.all(segmentation_result == result):
            return segmentation_result
        result = segmentation_result
    return segmentation_result
