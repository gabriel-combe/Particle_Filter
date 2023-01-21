from Particle import Particle, ConstAccelParticle2D
from typing import List

# Class to hold multiple particles
# for multi target tracking
class Tracker(object):
    def __init__(self, particle_struct: Particle= ConstAccelParticle2D, track_dim: int =1, weight: float =1.):
        # Number of targets we want to track
        self.track_dim = track_dim 

        # Type of particle we want to use
        self.particle_struct = particle_struct

        # Weight of the tracker
        # Tells how good the tracker model the behaviour of the real system
        self.weight = weight

        # List of Particle (one for each targets)
        self.trackers: List[self.particle_struct] = [self.particle_struct() for _ in range(self.track_dim)]
    
    # Create a tracker which has uniformly distributed particles in a given range
    def create_uniform_particles(self, ranges: list) -> object:
        for i, particle in enumerate(self.trackers):
            particle.create_uniform_particle(ranges[i])
        return self
    
    # Create a tracker which has normally distributed particles
    # for a given mean (init_poses) and standard deviation (stds)
    def create_gaussian_particles(self, init_poses: list, stds: list) -> object:
        for i, particle in enumerate(self.trackers):
            particle.create_gaussian_particle(init_poses[i], stds[i])
        return self
    
    # Predict the tracker's next state 
    # by calling the motion model of each particles in the tracker
    def predict_tracker(self, u: list, Q_control: list, Q_model: list, dt: float =1.) -> None:
        for i, particle in enumerate(self.trackers):
            particle.motion_model(u[i], Q_control[i], Q_model[i], dt)

    # Update the tracker's belief (weight) of representing the real system
    # by calling the measurement model of each particles in the tracker.
    # Given observations (zs) and standard deviation (Rs)
    # Can provide additionnal arguments (args) if needed
    def update_tracker(self, zs: list, Rs: list, args=()) -> None:
        for i, particle in enumerate(self.trackers):
            self.weight *= particle.measurement_model(zs[i], Rs[i], args)
        self.weight += 1.e-12
    
    # + operator overload
    def __add__(self, other: object):
        assert self.track_dim == other.track_dim

        new_tracker = self.__class__(self.particle_struct, self.track_dim)

        for i in range(self.track_dim):
            new_tracker.trackers[i] = self.trackers[i] + other.trackers[i]

        return new_tracker

    # += operator overload
    def __iadd__(self, other: object):
        assert self.track_dim == other.track_dim

        for i in range(self.track_dim):
            self.trackers[i] += other.trackers[i]

        return self

    # - operator overload
    def __sub__(self, other: object):
        assert self.track_dim == other.track_dim

        new_tracker = self.__class__(self.particle_struct, self.track_dim)

        for i in range(self.track_dim):
            new_tracker.trackers[i] = self.trackers[i] - other.trackers[i]

        return new_tracker

    # -= operator overload
    def __isub__(self, other: object):
        assert self.track_dim == other.track_dim

        for i in range(self.track_dim):
            self.trackers[i] -= other.trackers[i]

        return self
    
    # pow() (**) operator overload
    def __pow__(self, exposant: object):
        assert isinstance(exposant, (int, float))

        new_tracker = self.__class__(self.particle_struct, self.track_dim)

        for i in range(self.track_dim):
            new_tracker.trackers[i] = self.trackers[i] ** exposant

        return new_tracker
    
    # **= operator overload
    def __ipow__(self, exposant: object):
        assert isinstance(exposant, (int, float))

        for i in range(self.track_dim):
            self.trackers[i] **= exposant

        return self
    
    # * operator overload
    def __mul__(self, mul: object):
        assert isinstance(mul, (int, float))

        new_tracker = self.__class__(self.particle_struct, self.track_dim)

        for i in range(self.track_dim):
            new_tracker.trackers[i] = self.trackers[i] * mul

        return new_tracker
    
    # inverse * operator overload
    def __rmul__(self, mul: object):
        assert isinstance(mul, (int, float))

        new_tracker = self.__class__(self.particle_struct, self.track_dim)

        for i in range(self.track_dim):
            new_tracker.trackers[i] = mul * self.trackers[i]

        return new_tracker

    # *= operator overload
    def __imul__(self, mul: object):
        assert isinstance(mul, (int, float))

        for i in range(self.track_dim):
            self.trackers[i] *= mul

        return self

    # / operator overload
    def __truediv__(self, denom: object):
        assert isinstance(denom, (int, float))

        new_tracker = self.__class__(self.particle_struct, self.track_dim)

        for i in range(self.track_dim):
            new_tracker.trackers[i] = self.trackers[i] / denom

        return new_tracker

    # /= operator overload
    def __itruediv__(self, denom: object):
        assert isinstance(denom, (int, float))

        for i in range(self.track_dim):
            self.trackers[i] /= denom

        return self

    # Print function overload
    def __str__(self) -> str:
        string_format = ''
        for attrib in self.particle_struct().__dict__:
            if attrib == 'particle_dim': continue
            string_format += f'{attrib} = [ '
            for particle in self.trackers:
                string_format += f'{particle.__getattribute__(attrib):.4f}  '
            string_format += ']\n'
        return string_format