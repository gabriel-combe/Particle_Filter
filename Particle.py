import numpy as np
import scipy.stats as ss
from typing import Optional
from ParticleTemplate import Particle

##################################
### SimplePosHeadingParticle2D ###

class SimplePosHeadingParticle2D(Particle):
    default_ranges: list = [[0, 20], [0, 20], [0, 2*np.pi]]

    def __init__(self, 
                x: float =0., 
                y: float =0., 
                hdg: float =0.):

        self.particle_dim = 3

        self.x = x
        self.y = y
        self.hdg = hdg
    
    def create_uniform_particles(self,ranges: np.ndarray =default_ranges) -> Particle:
        super().create_uniform_particle(ranges)
        self.hdg %= 2 * np.pi
        return self
    
    def create_gaussian_particles(self, init_pos: np.ndarray, std: np.ndarray) -> Particle:
        super().create_gaussian_particle(init_pos, std)
        self.hdg %= 2 * np.pi
        return self


##################################
###### ConstAccelParticle2D ######

# 2D Particle with position, velocity and a constant acceleration
class ConstAccelParticle2D(Particle):

    def __init__(self):
        self.particle_dim = 6
    
     # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        return super().create_uniform_particles(N, track_dim, ranges)

    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        return super().create_gaussian_particles(N, track_dim,  init_pos, std)
    
    # Constant acceleration prediction model using simple equations of motion.
    # We add random noise to acceleration
    # to model non constant acceleration system.
    # We dot not have control input in this model.
    # TODO add modularity to handle use of control input in  the same Particle class
    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray], Q_control: Optional[np.ndarray]) -> np.ndarray:
        N = particles.shape[0]
        track_dim = particles.shape[1]

        # X positions
        particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt + np.random.randn(N, track_dim) * Q_model[:, 0]
        # X velocities
        particles[:, :, 1] += particles[:, :, 2] * dt + np.random.randn(N, track_dim) * Q_model[:, 1]
        # X accelerations
        particles[:, :, 2] += np.random.randn(N, track_dim) * Q_model[:, 2]
        
        # Y positions
        particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt + np.random.randn(N, track_dim) * Q_model[:, 3]
        # Y velocities
        particles[:, :, 4] += particles[:, :, 5] * dt + np.random.randn(N, track_dim) * Q_model[:, 4]
        # Y accelerations
        particles[:, :, 5] += np.random.randn(N, track_dim) * Q_model[:, 5]

        return particles
    
    # Measurement model assume we give him 
    # xy positions and velocities.
    # Can use additionnal arguments (args)
    def measurement_model(self, particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray, args=()) -> np.ndarray:
        pos_proba = ss.norm(0., R[:, 0]).pdf(particles[:, :, 0] - z[:, 0]) * ss.norm(0., R[:, 1]).pdf(particles[:, :, 3] - z[:, 1])
        vel_proba = ss.norm(0., R[:, 2]).pdf(particles[:, :, 1] - z[:, 2]) * ss.norm(0., R[:, 3]).pdf(particles[:, :, 4] - z[:, 3])
        return np.sum(pos_proba * vel_proba, axis=1)


##################################
####### ConstVelParticle2D #######

# init_pos = np.repeat([[[1, 1, 1, 1, 1, 1], [.1, .1, .1, .1, .1, .1]]], 2, axis=0)
# p = ConstAccelParticle2D(5, 2).create_gaussian_particles(init_pos[:, 0], init_pos[:, 1])
# print(p.samples)
# p.motion_model(None, None, np.array([[1, 1, 1, 1, 1, 1], [.1, .1, .1, .1, .1, .1]]))
# print(p.samples)
# p.measurement_model(np.array([[1, 1, .1, .1], [2, 2, .2, .2]]), np.array([[.1, .1, .1, .1], [.2, .2, .2, .2]]))
# print(p.weights)