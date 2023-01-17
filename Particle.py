import numpy as np
import scipy.stats as ss
from typing import Optional

class Particle(object):
    def __init__(self):
        self.particle_dim: int
    
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

    def __init__(self, x: float =0., y: float =0., hdg: float =0.):
        self.particle_dim = 3

        self.x = x
        self.y = y
        self.hdg = hdg
    
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
    default_Q_model: np.ndarray = np.ones(6)
    
    def __init__(self, x: float =0., vx: float =0., ax: float =0., y: float =0., vy: float =0., ay: float =0., weight: float =1., Q_model: np.ndarray =default_Q_model, R: np.ndarray =None):
        self.particle_dim = 6

        self.x = x
        self.vx = vx
        self.ax = ax
        self.y = y
        self.vy = vy
        self.ay = ay

        self.weight = weight

        self.Q_model = Q_model
        self.R = R
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray =default_ranges) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)
    
    def motion_model(self, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None, dt: float =1.) -> None:

        # X position
        self.x += -.5 * self.ax * dt**2 + self.vx * dt + np.random.randn() * self.Q_model[0]
        # X velocity
        self.vx += self.ax * dt + np.random.randn() * self.Q_model[1]
        # X acceleration
        self.ax += np.random.randn() * self.Q_model[2]
        
        # Y position
        self.y += -.5 * self.ay * dt**2 + self.vy * dt + np.random.randn() * self.Q_model[3]
        # Y velocity
        self.vy += self.ay * dt + np.random.randn() * self.Q_model[4]
        # Y acceleration
        self.ay += np.random.randn() * self.Q_model[5]
    
    def measurement_model(self, z: np.ndarray) -> None:
        pos_error = np.sqrt((self.x - z[0])**2 + (self.y - z[1])**2)
        vel_error = np.sqrt((self.vx - z[2])**2 + (self.vy - z[3])**2)
        self.weight *= ss.norm(0, self.R[0]).pdf(pos_error)
        self.weight *= ss.norm(0, self.R[2]).pdf(vel_error)


##################################
####### ConstVelParticle2D #######

class ConstVelParticle2D(Particle):
    default_ranges: np.ndarray = np.array([[0, 20], [-1, 1], [0, 20], [-1, 1]])
    
    def __init__(self, x: float =0., vx: float =0., y: float =0., vy: float =0.):
        self.particle_dim = 4

        self.x = x
        self.vx = vx
        self.y = y
        self.vy = vy
    
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray =default_ranges) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)