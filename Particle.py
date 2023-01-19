import numpy as np
import scipy.stats as ss
from typing import Optional

class Particle(object):
    def __init__(self):
        self.particle_dim: int =0
    
    def create_uniform_particle(self, ranges: list) -> object:
        offset = 0
        for i, attrib in enumerate(self.__dict__.keys()):
            if attrib in ['particle_dim', 'Q_model', 'R', 'weight']:
                offset += 1
                continue
            self.__setattr__(attrib, np.random.uniform(ranges[i - offset][0], ranges[i - offset][1]))
        return self

    def create_gaussian_particle(self, init_pos: list, std: list) -> object:
        offset = 0
        for i, attrib in enumerate(self.__dict__.keys()):
            if attrib in ['particle_dim', 'Q_model', 'R', 'weight']: 
                offset += 1
                continue
            self.__setattr__(attrib, init_pos[i - offset] + (np.random.randn() * std[i - offset]))
        return self

    def __str__(self) -> str:
        string_format = ''
        for attrib in self.__dict__:
            if attrib in ['particle_dim', 'Q_model', 'R', 'weight']: continue
            string_format += f'{attrib} = {self.__dict__[attrib]}\n'
        return string_format
            


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
    
    def create_uniform_particle(self,ranges: list =default_ranges) -> Particle:
        super().create_uniform_particle(ranges)
        self.hdg %= 2 * np.pi
        return self
    
    def create_gaussian_particle(self, init_pos: list, std: list) -> Particle:
        super().create_gaussian_particle(init_pos, std)
        self.hdg %= 2 * np.pi
        return self


##################################
###### ConstAccelParticle2D ######

class ConstAccelParticle2D(Particle):
    default_ranges: list = [[0, 20], [-1, 1], [-0.1, 0.1], [0, 20], [-1, 1], [-0.1, 0.1]]
    default_Q_model: list = [1., 1., 1., 1., 1., 1.]
    
    def __init__(self, 
                x: float =0., 
                vx: float =0., 
                ax: float =0., 
                y: float =0., 
                vy: float =0., 
                ay: float =0., 
                weight: float =1., 
                Q_model: list =default_Q_model, 
                R: np.ndarray =None):

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
    
    def create_uniform_particle(self, ranges: list =default_ranges) -> Particle:
        return super().create_uniform_particle(ranges)

    
    def motion_model(self, u: Optional[list] =None, Q_control: Optional[list] =None, dt: float =1.) -> None:

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
    
    def measurement_model(self, z: list) -> None:
        pos_error = np.sqrt(((self.x - z[0])/self.R[0])**2 + ((self.y - z[1])/self.R[1])**2)
        vel_error = np.sqrt(((self.vx - z[2])/self.R[2])**2 + ((self.vy - z[3])/self.R[3])**2)
        self.weight = ss.norm(0., 1.).pdf(pos_error) * ss.norm(0., 1.).pdf(vel_error)
        # self.weight += 1.e-12


##################################
####### ConstVelParticle2D #######

class ConstVelParticle2D(Particle):
    default_ranges: list = [[0, 20], [-1, 1], [0, 20], [-1, 1]]
    
    def __init__(self, 
                x: float =0., 
                vx: float =0., 
                y: float =0., 
                vy: float =0.):

        self.particle_dim = 4

        self.x = x
        self.vx = vx
        self.y = y
        self.vy = vy
    
    def create_uniform_particle(self, ranges: list =default_ranges):
        return super().create_uniform_particle(ranges)