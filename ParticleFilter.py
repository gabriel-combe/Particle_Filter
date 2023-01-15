import numpy as np
from typing import Optional
from copy import deepcopy
from ResampleMethods import systematic_resample
from Particle import Particle, ConstAccelParticle2D, SimplePosHeadingParticle2D
from Models import default_motion_model, default_measurement_model

# Class for the particle filter object
class ParticleFilter(object):
    def __init__(self, 
                N: int, 
                particle_struct=SimplePosHeadingParticle2D, 
                track_dim: int =1,
                control_dim: int =0,
                init_pos: np.ndarray =None,
                ranges: tuple =None,
                motion_model_fn=default_motion_model,
                measurement_model_fn=default_measurement_model,
                Q_motion: Optional[np.ndarray] =None,
                Q_control: Optional[np.ndarray] =None,
                R: Optional[np.ndarray] =None,
                resample_method_fn=systematic_resample):

        self.N = N
        self.track_dim = track_dim
        self.control_dim = control_dim
        self.particle_struct = particle_struct
        self.state_dim = self.particle_struct().particle_dim

        self.motion_model = motion_model_fn
        self.measurement_model = measurement_model_fn

        self.Q_motion = Q_motion
        if self.Q_motion is None:
            self.Q_motion = np.ones((self.track_dim, self.state_dim))
        elif np.isscalar(self.Q_motion):
            self.Q_motion = np.ones((self.track_dim, self.state_dim)) * self.Q_motion
        
        self.Q_control = Q_control
        if self.control_dim != 0:
            if self.Q_control is None:
                self.Q_control = np.ones((self.track_dim, control_dim))
            elif np.isscalar(self.Q_control):
                self.Q_control = np.ones((self.track_dim, control_dim)) * self.Q_control
        
        self.R = R
        if self.R is None:
            self.R = np.eye(self.track_dim)
        elif np.isscalar(self.R):
            self.R = np.eye(self.track_dim) * self.R
        
        self.resample_method = resample_method_fn

        self.particles: np.ndarray = None

        if init_pos is not None:
            self.particles = self.particle_struct().create_gaussian_particles(init_pos[0], init_pos[1], self.N, self.track_dim)
        elif ranges is not None:
            self.particles = self.particle_struct().create_uniform_particles(self.N, self.track_dim, ranges)
        else:
            self.particles = self.particle_struct().create_uniform_particles(self.N, self.track_dim)


        self.weights: np.ndarray = np.ones(N) / N
        self.mu: float = 0.
        self.sigma: float = 0.

    # Predict next particles state (prior) 
    # using motion_model
    # and control_model with input u.
    def predict(self, 
                u: Optional[np.ndarray] =None, 
                dt: Optional[float] =1.) -> None:

        self.particles = self.motion_model(self.particles, self.Q_motion, dt, u, self.Q_control)

    # Update the particles state by integrating new measurements
    def update(self, z: np.ndarray, args=()) -> None:
        
        self.weights = self.measurement_model(self.particles, self.weights, z, self.R, *args)

        self.weights += 1.e-12
        self.weights /= sum(self.weights)

    # Computation of the mean and variance of the particles (estimate)
    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        
        position = self.particles[:]
        self.mu = np.average(position, weights=self.weights, axis=0)
        self.sigma = np.average((position - self.mu)**2, weights=self.weights, axis=0)
        return (self.mu, self.sigma)

    # Compute the effective N value
    def neff(self) -> float:
        return 1. / np.sum(np.square(self.weights))

    # Perform resample 
    def resample(self, fraction: Optional[float] =1./4.) -> None:
        if self.neff() < self.N * fraction:
            indexes = self.resample_method(self.weights)
            self.particles[:] = self.particles[indexes]
            self.weights.resize(self.N)
            self.weights.fill(1/self.N)

