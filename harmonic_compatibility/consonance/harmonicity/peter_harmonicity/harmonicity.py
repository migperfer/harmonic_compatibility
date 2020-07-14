import numpy as np
eps = np.finfo(float).eps


def pc_harmonics(pc, n_harmonics=12):
    """
    Get the pitch classes harmonics given a pitch class set
    :param pc: The pitch class set
    :param n_harmonics: The number of harmonics to calculate
    :return: The pitch class sets with the harmonics
    """
    pc = np.array(pc)
    return np.mod(pc[:, np.newaxis] + 12 * np.log2(np.arange(1, n_harmonics + 1)), 12)


def pc_distance(pc1, pc2):
    """
    Pitch class distance between pc1 and pc2. One of them can be a numpy array.
    :param pc1: Pitch class 1
    :param pc2: Pitch class 2
    :return: pitch class distance/distances.
    """
    t_min = np.minimum(abs(pc1 - pc2[:, np.newaxis]), 12 - abs(pc1 - pc2[:, np.newaxis]))
    return t_min


def h_contribution(pc, l, px, sigma=0.0683):
    """
    Harmonic contribution to pitch class pc, from pitch class px with level l.
    :param pc: The pitch class to which contributes
    :param l: The loudness level [0,1]
    :param px: The pitch class that contributes to certain sensation
    :return: The harmonic contribution
    """
    const = l / (sigma * np.sqrt(2 * np.pi))
    distances = pc_distance(pc, px)
    t_return = const * np.exp(-0.5 * np.square(distances / sigma))
    return np.sum(t_return)


def cosine_similarity(X, Y):
    """
    Cosine similarity between vectors X and Y
    :param X: First vector
    :param Y: Second vector
    :return: Cosine similarity
    """
    num = np.sum(X * Y)
    den = np.sqrt(np.sum(np.square(X))) * np.sqrt(np.sum(np.square(Y)))
    return num / den


def perceptual_weight(pc, x, rho=0.75, sigma=0.0683):
    """
    Returns the perceptual weight for pc, given the set x of pitches. Pitch class spectrum
    :param pc: The pitch class to calculate the weight. A float in the range (0, 11]
    :param x: Pitch class/pitch classes that contribute to pc. Can be either a single value (float/int) or an iterable (ndarray/list/tuple)
    :return: The perceptual weight for class pc
    """
    if isinstance(x, (int, float)):
        x = np.array([x])
    x_harmonics = pc_harmonics(x)
    plane_harmonics = x_harmonics.reshape(len(x) * 12)
    n_repetitions = len(x)
    weights_harmonics = np.repeat([np.power(np.arange(1, 13), rho)],
                                  n_repetitions,
                                  axis=0).reshape(len(x) * 12, 1) / (sigma * np.sqrt(2 * np.pi))
    distances = pc_distance(pc, plane_harmonics)
    exppart = np.exp(-0.5 * np.square(distances / sigma))

    weighted_contribution = np.sum(exppart * weights_harmonics, axis=0)
    return weighted_contribution


def milne_pc_spectrum(X):
    """
    Calculates the milne pc spectrum.
    :param X: The pitch class set.
    :return:
    """
    X = np.array(X)
    pc_spec = perceptual_weight(np.linspace(0, 12, 1200), X)
    pc_spec /= 100
    return pc_spec


def spectral_distance(X, Y):
    """
    Spectral distance as defined in:
    Harrison, P., & Pearce, M. T. (2018).
    An energy-based generative sequence model for testing sensory theories of Western harmony.
    arXiv preprint arXiv:1807.00790.
    :param X: milne_pc_spectrum 1
    :param Y: milne_pc_spectrum 2
    :return: The spectral distance
    """
    return 1 - cosine_similarity(X, Y)


def virtual_pitch_class_spectrum(pc, X, normalized=True):
    pc_c_spectrum = milne_pc_spectrum(pc_harmonics(pc))
    num = spectral_distance(pc_c_spectrum, X)
    if normalized:
        template = milne_pc_spectrum(pc_harmonics(0))
        all_pob = np.convolve(X, np.concatenate([template, template]), mode='valid')
        all_pob = all_pob / (np.linalg.norm(X) * np.linalg.norm(template))
        all_pob = all_pob[:-1]
        den = np.sum(all_pob)
    else:
        den = 1
    return num / np.sum(den)


def harmonicity(X):
    template = milne_pc_spectrum([0])
    all_pob = np.correlate(X, np.concatenate([template, template]), mode='valid')
    all_pob = all_pob / (np.linalg.norm(X) * np.linalg.norm(template))
    all_pob = all_pob[:-1]
    q_normalized = all_pob / (np.sum(all_pob))
    return np.sum(q_normalized * np.log2(eps + (q_normalized * len(all_pob))))