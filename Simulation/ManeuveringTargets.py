import numpy as np
from State import State

class ManeuveringTarget(object): 
    def __init__(self, state0: State) -> None:
        self.state = state0
        
        self.cmd_vel = state0.vel
        self.cmd_hdg = state0.hdg
        self.vel_step = 0
        self.hdg_step = 0
        self.vel_delta = 0
        self.hdg_delta = 0
    
    def update(self) -> State:
        vx = self.state.vel * np.cos(np.radians(90-self.state.hdg))
        vy = self.state.vel * np.sin(np.radians(90-self.state.hdg))
        self.state.x += vx
        self.state.y += vy
        
        if self.hdg_step > 0:
            self.hdg_step -= 1
            self.state.hdg += self.hdg_delta

        if self.vel_step > 0:
            self.vel_step -= 1
            self.state.vel += self.vel_delta

        return self.state

    def set_commanded_heading(self, hdg_degrees: float, steps: int) -> None:
        self.cmd_hdg = hdg_degrees
        self.hdg_delta = self.state.angle_between(self.cmd_hdg) / steps

        if abs(self.hdg_delta) > 0:
            self.hdg_step = steps
        else:
            self.hdg_step = 0
         
    def set_commanded_speed(self, speed: float, steps: int) -> None:
        self.cmd_vel = speed
        self.vel_delta = (self.cmd_vel - self.state.vel) / steps

        if abs(self.vel_delta) > 0:
            self.vel_step = steps
        else:
            self.vel_step = 0    