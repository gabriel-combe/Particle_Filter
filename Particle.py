class Particle(object):
    state_dim: int

    def __init__(self) -> None:
        pass
    pass

class ConstAccelParticle2D(Particle):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 6
    pass