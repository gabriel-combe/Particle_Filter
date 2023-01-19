from Particle import Particle, ConstAccelParticle2D

class Tracker(object):
    def __init__(self, particle_struct: Particle, track_dim: int =1, weight: float =1.):
        self.track_dim = track_dim

        self.particle_struct = particle_struct

        self.weight = weight

        self.trackers = [self.particle_struct() for _ in range(self.track_dim)]
    
    def create_uniform_particles(self, ranges: list):
        for particle in self.trackers:
            particle.create_uniform_particles(ranges)
        

tracker = Tracker(ConstAccelParticle2D, 10)
tracker.create_uniform_particles()
print(tracker.trackers)
