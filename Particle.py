import numpy as np
from typing import Optional

class Particle(object):
    state_dim: int

    def __init__(self) -> None:
        pass
    
    def create_uniform_particles(self, ranges: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = np.empty((N, track_dim, self.state_dim))
        for i in range(self.state_dim):
            particles[:, :, i] = np.random.uniform(ranges[i][0], ranges[i][1], size=(N, track_dim))
        return particles

    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = np.empty((N, track_dim, self.state_dim))
        for i in range(self.state_dim):
            particles[:, :, i] = init_pos[i] + (np.random.randn(N, track_dim) * std[i])
        return particles


class SimplePosHeadingParticle2D(Particle):
    def __init__(self, x: float =0, y: float =0, hdg: float =0) -> None:
        super().__init__()
        self.state_dim = 3

        self.x = x
        self.y = y
        self.hdg = hdg
    
    def create_uniform_particles(self, N: int, track_dim: int,  ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] =((0, 20), (0, 20), (0, 2*np.pi))) -> np.ndarray:
        particles = super().create_uniform_particles(ranges, N, track_dim)
        particles[:, :, 2] %= 2 * np.pi
        return particles
    
    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray, N: int, track_dim: int) -> np.ndarray:
        particles = super().create_gaussian_particles(init_pos, std, N, track_dim)
        particles[:, :, 2] %= 2 * np.pi
        return particles

class ConstAccelParticle2D(Particle):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 6
    pass

particles = SimplePosHeadingParticle2D()
print(particles.create_uniform_particles(4, 1, ((0, 20), (0, 20), (0, 2*np.pi))))
print(particles.create_gaussian_particles(np.array([0, 0, 0]), np.array([.1, .1, .3]), 4, 1))

