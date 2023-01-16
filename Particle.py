import numpy as np

class Particle(object):
    def __init__(self):
        self.state_dim: int
        pass
    
    def create_uniform_particles(self, ranges: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        pass

    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        pass


class SimplePosHeadingParticle2D(object):
    def __init__(self):
        self.particle_dim = 3
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: tuple[tuple, tuple, tuple] =((0, 20), (0, 20), (0, 2*np.pi))) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = np.random.uniform(ranges[i][0], ranges[i][1], size=(N, track_dim))
        particles[:, :, 2] %= 2 * np.pi
        return particles
    
    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = init_pos[i] + (np.random.randn(N, track_dim) * std[i])
        particles[:, :, 2] %= 2 * np.pi
        return particles

class ConstAccelParticle2D(object):
    def __init__(self):
        self.particle_dim = 6
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: tuple[tuple, tuple, tuple, tuple, tuple, tuple] = ((0, 20), (-1, 1), (-0.1, 0.1), (0, 20), (-1, 1), (-0.1, 0.1))) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = np.random.uniform(ranges[i][0], ranges[i][1], size=(N, track_dim))
        return particles

    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = init_pos[i] + (np.random.randn(N, track_dim) * std[i])
        return particles

