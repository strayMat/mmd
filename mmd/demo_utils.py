import numpy as np
from sklearn.utils.validation import check_random_state
from matplotlib import cm

# Utils #


TAB_COLORS = [cm.tab20(i) for i in range(20)]
untreated_color = TAB_COLORS[2]
treated_color = TAB_COLORS[0]
COLOR_MAPPING = {0: untreated_color, 1: treated_color}
LABEL_MAPPING = {0: "Control", 1: "Treated"}


def generate_rotation(theta: float = None, random_state=None):
    """Generate a random rotation matrix for the given dimensionality and angle.

    Parameters
    ----------
    dim : int
        Dimensionality of the rotation matrix.
    theta : float
        Angle of rotation in radians.
    random_state : int or RandomState
        Random state for the random number generator.

    Returns
    -------
    rotation : ndarray
        Rotation matrix.
    """
    generator = check_random_state(random_state)
    if theta is None:
        theta = generator.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array(((c, -s), (s, c)))

    return rotation


def generate_gaussian_blobs(
    N: int, overlap: float = 1, sigma=np.array([2, 5]), population_ratio=0.5, seed=None
):
    generator = check_random_state(seed)

    sigma_ = np.diag(sigma)
    rotation = generate_rotation(theta=1, random_state=generator)

    center = np.array([overlap, 0])
    center_rotated_1 = rotation.dot(center)
    center_rotated_0 = rotation.dot(-center)
    sigma_rotated = rotation.dot(sigma_).dot(rotation.T)
    X_control = generator.multivariate_normal(
        mean=center_rotated_0,
        cov=sigma_rotated,
        size=int((1 - population_ratio) * N),
    )
    X_treated = generator.multivariate_normal(
        mean=center_rotated_1,
        cov=sigma_rotated,
        size=int(population_ratio * N),
    )
    return X_control, X_treated
