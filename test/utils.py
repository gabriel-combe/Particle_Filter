import numpy as np

def create_uniform_particles(x_range: tuple[float, float], y_range: tuple[float, float], hdg_range: tuple[float, float], N: int) -> np.ndarray:
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(hdg_range[0], hdg_range[1], size=N) % 2*np.pi
    return particles

def create_gaussian_particles(mean: tuple[float, float, float], std: tuple[float, float, float], N: int) -> np.ndarray:
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (np.random.randn(N) * std[2]) % 2*np.pi
    return particles

def neff(weights :np.ndarray) -> float:
    return 1. / np.sum(np.square(weights))

def resampling_from_index(particles :np.ndarray, weights :np.ndarray, indexes :np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1/len(weights))
    return particles, weights