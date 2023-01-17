import numpy as np

class Particle(object):
    def __init__(self):
        self.particle_dim: int
        pass
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = np.random.uniform(ranges[i][0], ranges[i][1], size=(N, track_dim))
        return particles

    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = np.empty((N, track_dim, self.particle_dim))
        for i in range(self.particle_dim):
            particles[:, :, i] = init_pos[i] + (np.random.randn(N, track_dim) * std[i])
        return particles


##################################
### SimplePosHeadingParticle2D ###

class SimplePosHeadingParticle2D(Particle):
    default_ranges: np.ndarray = np.array([[0, 20], [0, 20], [0, 2*np.pi]])

    def __init__(self):
        self.particle_dim = 3
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray =default_ranges) -> np.ndarray:
        particles = super().create_uniform_particles(N, track_dim, ranges)
        particles[:, :, 2] %= 2 * np.pi
        return particles
    
    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = super().create_gaussian_particles(init_pos, std, N, track_dim)
        particles[:, :, 2] %= 2 * np.pi
        return particles


##################################
###### ConstAccelParticle2D ######

class ConstAccelParticle2D(Particle):
    default_ranges: np.ndarray = np.array([[0, 20], [-1, 1], [-0.1, 0.1], [0, 20], [-1, 1], [-0.1, 0.1]])
    
    def __init__(self):
        self.particle_dim = 6
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray =default_ranges) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)

    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        return super().create_gaussian_particles(init_pos, std, N, track_dim)

