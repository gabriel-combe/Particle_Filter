import numpy as np
import scipy.stats as ss
from typing import Optional
from ParticleTemplate import Particle

##################################
### SimplePosHeadingParticle2D ###

class SimplePosHeadingParticle2D(Particle):
    def __init__(self, 
                x: float =0., 
                y: float =0., 
                hdg: float =0.):

        self.particle_dim = 3

        self.x = x
        self.y = y
        self.hdg = hdg
    
    # Create particles which are uniformly distributed in given ranges
    def create_uniform_particles(self, N: int, track_dim: int, ranges: np.ndarray) -> np.ndarray:
        particles = super().create_uniform_particles(N, track_dim, ranges)
        particles[:, :, 2] %= 2 * np.pi
        return particles
    
    # Create particles from a gaussian distribution
    # with mean (init_pos) and standard deviation (std)
    def create_gaussian_particles(self, N: int, track_dim: int, init_pos: np.ndarray, std: np.ndarray) -> np.ndarray:
        particles = super().create_gaussian_particles(N, track_dim,  init_pos, std)
        particles[:, :, 2] %= 2 * np.pi
        return particles

    def motion_model(self, particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray], Q_control: Optional[np.ndarray]) -> np.ndarray:
        N = particles.shape[0]
        track_dim = particles.shape[1]

        particles[:, :, 2] += (u[:, 0] + (np.random.randn(N, track_dim) * Q_control[:, 0])) % 2*np.pi

        distance = (u[:, 1] * dt) + (np.random.randn(N, track_dim) * Q_control[:, 1])

        particles[:, :, 0] += np.cos(particles[:, :, 2]) * distance
        particles[:, :, 1] += np.sin(particles[:, :, 2]) * distance

        return particles

    def measurement_model(self, particles: np.ndarray, z: np.ndarray, R: np.ndarray, args=()) -> np.ndarray:
        proba = 1.
        for i in range(particles.shape[1]):
            for k, landmark in enumerate(args[0]):
                distance = np.linalg.norm(particles[:, i, :2] - landmark, axis=1)
                proba *= ss.norm(distance, R[i]).pdf(z[i, k])
            proba *= proba

        return proba


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
    def measurement_model(self, particles: np.ndarray, z: np.ndarray, R: np.ndarray, args=()) -> np.ndarray:
        pos_proba = ss.norm(0., R[:, 0]).pdf(particles[:, :, 0] - z[:, 0]) * ss.norm(0., R[:, 1]).pdf(particles[:, :, 3] - z[:, 1])
        vel_proba = ss.norm(0., R[:, 2]).pdf(particles[:, :, 1] - z[:, 2]) * ss.norm(0., R[:, 3]).pdf(particles[:, :, 4] - z[:, 3])
        return np.sum(pos_proba * vel_proba, axis=1)


##################################
####### ConstVelParticle2D #######