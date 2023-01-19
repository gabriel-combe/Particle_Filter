from Particle import Particle, ConstAccelParticle2D

class Tracker(object):
    def __init__(self, particle_struct: Particle= ConstAccelParticle2D, track_dim: int =1, weight: float =1.):
        self.track_dim = track_dim

        self.particle_struct = particle_struct

        self.weight = weight

        self.trackers = [self.particle_struct() for _ in range(self.track_dim)]
    
    def create_uniform_particles(self) -> None:
        for particle in self.trackers:
            particle.create_uniform_particle()
    
    def create_uniform_particles(self, ranges: list) -> None:
        for particle in self.trackers:
            particle.create_uniform_particle(ranges)
    
    def create_gaussian_particles(self, init_poss: list, stds: list) -> None:
        for i, particle in enumerate(self.trackers):
            particle.create_gaussian_particle(init_poss[i], stds[i])
        
    def __str__(self) -> str:
        string_format = ''
        for attrib in self.particle_struct().__dict__:
            if attrib in ['particle_dim', 'Q_model', 'R', 'weight']: continue
            string_format += f'{attrib} = [ '
            for particle in self.trackers:
                string_format += f'{particle.__getattribute__(attrib):.3f}  '
            string_format += ']\n'
        return string_format

trackers = Tracker(ConstAccelParticle2D, 2)
trackers.create_gaussian_particles([[5, 1, 1, 5, 1, 2], [0, 1, 1, 0, 1, 2]], [[.2, .3, .2, .2, .3, .2], [.2, .3, .2, .2, .3, .2]])
print(trackers)
