class State(object):
    """Represents the state vector"""
    def __init__(self, x: float =0, y: float =0, vel: float =0, hdg: float =0) -> None:
        self.x = x
        self.y = y
        self.vel = vel
        self.hdg = hdg

    def angle_between(self, heading) -> float:
        return min(self.hdg-heading, self.hdg-heading+360, self.hdg-heading-360, key=abs)

    def __add__(self, o):
        return State(self.x + o.x, self.y + o.y, self.vel + o.vel, (self.hdg + o.hdg) % 360)

    def __str__(self) -> str:
        return f'State: x = {self.x}    y = {self.y}    velocity = {self.vel}    heading = {self.hdg}'