import numpy as np
from typing import Optional
from ResampleMethods import systematic_resample
from Particle import Particle, SimplePosHeadingParticle2D, ConstAccelParticle2D

# Class for the particle filter object
class ParticleFilter(object):
    def __init__(self, 
                N: int, 
                particle_struct: Particle =SimplePosHeadingParticle2D, 
                track_dim: int =1,
                control_dim: int =0,
                init_pos: np.ndarray =None,
                ranges: np.ndarray =None,
                Q_motion: Optional[np.ndarray] =None,
                Q_control: Optional[np.ndarray] =None,
                R: Optional[np.ndarray] =None,
                resample_method_fn=systematic_resample):

        self.N = N
        self.track_dim = track_dim
        self.particle_struct = particle_struct
        self.state_dim = self.particle_struct().particle_dim
        self.control_dim = control_dim if control_dim != 0 else self.state_dim

        # Default control vector in case we do not provide control input
        # for the prediction step
        self.default_control = np.zeros((1, self.state_dim)).repeat(self.track_dim, axis=0)

        # If we don't give ranges, we use defaults ranges.
        # If we only give a list we repeat it for all targets
        self.ranges = ranges
        if self.ranges is None:
            self.ranges = np.repeat([np.repeat([[0., 1.]], self.state_dim, axis=0)], self.track_dim, axis=0)
        elif np.isscalar(self.ranges[0]):
            self.ranges = np.repeat([self.ranges], [self.track_dim], axis=0)
        
        # If we only give an array we repeat it for all targets
        self.init_pos = init_pos
        if self.init_pos is not None and np.isscalar(self.init_pos[0, 0]):
            self.init_pos = np.repeat([self.init_pos], self.track_dim, axis=0)

        # If we don't give Q_motion, we use defaults standard deviation.
        # If we only give a scalar we repeat it for all targets
        self.Q_motion = Q_motion
        if self.Q_motion is None:
            self.Q_motion = np.ones((self.track_dim, self.state_dim))
        elif np.isscalar(self.Q_motion):
            self.Q_motion = np.ones((self.track_dim, self.state_dim)) * self.Q_motion
        
        # If we don't give Q_control, we use defaults standard deviation.
        # If we only give a scalar we repeat it for all targets
        self.Q_control = Q_control
        if self.Q_control is None:
            self.Q_control = np.ones((self.track_dim, self.control_dim))
        elif np.isscalar(self.Q_control):
            self.Q_control = np.ones((self.track_dim, self.control_dim)) * self.Q_control
        
        # If we don't give R, we use defaults standard deviation.
        # If we only give a scalar we repeat it for all targets
        self.R = R
        if self.R is None:
            self.R = np.ones((self.track_dim, self.state_dim))
        elif np.isscalar(self.R):
            self.R = np.ones((self.track_dim, self.state_dim)) * self.R
        
        self.resample_method = resample_method_fn

        # Array of all the trackers and particles
        self.particles: np.ndarray = np.zeros((self.N, self.track_dim, self.state_dim))
        if init_pos is not None:
            self.particles = self.particle_struct().create_gaussian_particles(self.N, self.track_dim, init_pos[:, 0], init_pos[:, 1])
        else:
            self.particles = self.particle_struct().create_uniform_particles(self.N, self.track_dim, self.ranges)
        
        # Weights of each trackers
        self.weights: np.ndarray = np.ones(self.N)/self.N
        
        # Mean of all the particles for each targets
        self.mu: np.ndarray = np.zeros(self.state_dim)

        # Standard deviation of all the particles for each targets
        self.sigma: np.ndarray = np.zeros(self.state_dim)

    # Predict next state for each trackers (prior)
    def predict(self, 
                u: Optional[list] =None, 
                dt: Optional[float] =1.) -> None:
        
        self.particles = self.particle_struct().motion_model(self.particles, self.Q_motion, dt, u, self.Q_control)

    # Update each tracker belief with observations (z).
    # Can provide additionnal arguments (args) if needed
    def update(self, z: np.ndarray, args=()) -> None:
        self.weights *= self.particle_struct().measurement_model(self.particles, z, self.R, args)
        self.weights += 1.e-12
        self.weights /= np.sum(self.weights)

    # Computation of the mean and standard deviation
    # of the particles for each targets (estimate)
    def estimate(self) -> tuple[np.ndarray, np.ndarray]:
        
        self.mu = np.average(self.particles, weights=self.weights, axis=0)
        self.sigma = np.average((self.particles - self.mu)**2, weights=self.weights, axis=0)

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

    # Perform one pass of the particle filter
    def forward(self, z: np.ndarray, u: Optional[np.ndarray] =None, dt: float =1., fraction: float =1./4., args=(), verbose: int =0) -> None:
        self.predict(u, dt)
        self.update(z, args)
        self.resample(fraction)
        self.estimate()

        # for i in range(self.track_dim):
        #     if verbose >= 2:
        #         print(f'Mean Target {i+1}:\n\tposition {self.mu[i, ::3]}\n\tvelocity {self.mu[i, 1::3]}\n\tacceleration {self.mu[i, 2::3]}')
        #     if verbose >= 3:
        #         print(f'STD Target {i+1}:\n\tposition {self.sigma[i, ::3]}\n\tvelocity {self.sigma[i, 1::3]}\n\tacceleration {self.sigma[i, 2::3]}\n\n')
    
    # Loop over every observations
    # And perform a pass of the particle filter
    # for each observations
    def full_forward(self, measurements: np.ndarray, u: Optional[np.ndarray] =None, dt: float =1., fraction: float =1./4., args=(), verbose: int =0):
        for i, z in enumerate(measurements):
            if verbose >= 1:
                print(f'\nIterations {i+1}\n')

            self.forward(z, u[i], dt, fraction, args, verbose)