import numpy as np
import matplotlib.pyplot as plt
from State import State
from copy import deepcopy
from NoisySensor import NoisySensor
from ManeuveringTargets import ManeuveringTarget

def generate_data(steady_count, std) -> tuple[np.ndarray, np.ndarray]:
    t = ManeuveringTarget(State(vel=0.3))
    history = [deepcopy(t.state)]

    for _ in range(30):
        state = t.update()
        history.append(deepcopy(state))

    t.set_commanded_heading(310, 25)
    t.set_commanded_speed(1, 15)

    for _ in range(steady_count):
        state = t.update()
        history.append(deepcopy(state))

    ns = NoisySensor(std)
    history = np.array(history)
    zs = np.array([ns.sense(s) for s in history])
    return history, zs

sensor_std = 2.
track, zs = generate_data(50, sensor_std)
plt.figure()
plt.grid()
print(zs[0])
plt.scatter(*zip(*[(z.x, z.y) for z in zs]), alpha=.5)
plt.plot(*zip(*[(p.x, p.y) for p in track]), color='b', label='track')
plt.axis('equal')
plt.legend(loc=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Track vs Measurements')
plt.show()
