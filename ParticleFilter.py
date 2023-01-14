import numpy as np
from typing import Optional
from ResampleMethods import systematic_resample
from Particle import Particle, ConstAccelParticle2D

# Class for the particle filter object
class ParticleFilter(object):
    def __init__(self, 
                N: int, 
                particle_struct: Particle =ConstAccelParticle2D, 
                track_dim: int =1,
                control_dim: int =0,
                measurement_dim: int =0,
                motion_model: Optional[function] =None,
                Q_motion: Optional[np.ndarray] =None,
                control_model: Optional[function] =None,
                Q_control: Optional[np.ndarray] =None,
                measurement_model: function =None,
                R: Optional[np.ndarray] =None):

        self.N = N
        self.track_dim = track_dim
        self.control_dim = control_dim
        self.measurement_dim = measurement_dim
        self.particle_struct = particle_struct
        self.state_dim = self.particle_struct.state_dim
        
        self.motion_model = motion_model
        if self.motion_model is None:
            self.motion_model = np.eye(self.state_dim)
        
        self.control_model = control_model
        if self.control_model is None:
            self.control_model = np.eye(self.state_dim)

        self.Q_motion = Q_motion
        if self.Q_motion is None:
            self.Q_motion = np.eye(self.state_dim)
        elif np.isscalar(self.Q_motion):
            self.Q_motion = np.eye(self.state_dim) * self.Q_motion
        
        self.Q_control = Q_control
        if self.control_dim != 0:
            if self.Q_control is None:
                self.Q_control = np.eye(control_dim)
            elif np.isscalar(self.Q_control):
                self.Q_control = np.eye(control_dim) * self.Q_control
        
        self.measurement_model = measurement_model

        self.R = R
        if self.R is None:
            self.R = np.eye(self.measurement_dim)
        elif np.isscalar(self.R):
            self.R = np.eye(self.measurement_dim) * self.R

        self.particles: np.ndarray = None
        self.weights: np.ndarray = None
        self.mu: float = 0.
        self.sigma: float = 0.

    # Predict next particles state (prior) 
    # using motion_model
    # and control_model with input u.
    def predict(self, 
                u: Optional[np.ndarray] =None, 
                dt: Optional[float] =1.) -> None:

        if u is not None:
            self.particles = self.motion_model(self.particles, self.Q_motion, dt) + self.control_model(u, self.Q_control, dt)
        else:
            self.particles = self.motion_model(self.particles, self.Q_motion, dt)

    # Update the particles state by integrating new measurements
    def update(self, z: np.ndarray, args=()) -> None:

        self.weights = self.measurement_model(self.particles, self.weights, z, self.R, *args)

        self.weights += 1.e-12
        self.weights /= sum(self.weights)

    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        
        position = self.particles[:]
        self.mu = np.average(position, weights=self.weights, axis=0)
        self.sigma = np.average((position - self.mu)**2, weights=self.weights, axis=0)
        return (self.mu, self.sigma)

    def neff(self) -> float:
        return 1. / np.sum(np.square(self.weights))

    def resample(self, fraction: Optional[float] =1./4.) -> None:
        if self.neff(self.weights) < self.N * fraction:
            indexes = systematic_resample(self.weights)
            self.particles[:] = self.particles[indexes]
            self.weights.resize(self.N)
            self.weights.fill(1/self.N)

