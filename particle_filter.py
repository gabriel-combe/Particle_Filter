import scipy.stats
import numpy as np
from utils import *
from resampling import *
import matplotlib.pyplot as plt

def predict(particles :np.ndarray, u :tuple[float, float], Q :tuple[float, float], dt :float =1.) -> np.ndarray:
    N = len(particles)

    # Update heading
    particles[:, 2] += (u[0] + (np.random.randn(N) * Q[0])) %  2*np.pi

    # Move along the noisy heading
    distance = (u[1] * dt) + (np.random.randn(N) * Q[1])

    # Update position
    particles[:, 0] += np.cos(particles[:, 2]) * distance
    particles[:, 1] += np.sin(particles[:, 2]) * distance

    return particles

def update(particles :np.ndarray, weights :np.ndarray, z :np.ndarray, R :float, landmarks :np.ndarray) -> np.ndarray:
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, :2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])
    
    weights += 1.e-12
    weights /= sum(weights)
    return weights

def estimate(particles :np.ndarray, weights :np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    position = particles[:]
    mean = np.average(position, weights=weights, axis=0)
    var = np.average((position - mean)**2, weights=weights, axis=0)
    return (mean, var)

def particle_filter(N :int, controls :np.ndarray, measurements :np.ndarray, landmarks :np.ndarray, init_pos :tuple[tuple[float, float, float], tuple[float, float, float]] =None, sensor_std_err :float =.1, model_std_err :tuple[float, float] =(.2, .05)) -> np.ndarray:
    # Create particles and weights
    if init_pos is not None:
        particles = create_gaussian_particles(mean=init_pos[0], std=init_pos[1], N=N)
    else:
        particles = create_uniform_particles(x_range=(0, 20), y_range=(0, 20), hd_range=(0, 2*np.pi), N=N)

    weights = np.ones(N) / N
    
    # Create history to store each iteration
    history = [(particles.copy(), weights.copy(), None, None)]

    for u, z in zip(controls, measurements):
        # Predict the particles movement
        particles = predict(particles, u, Q=model_std_err)

        # Update the weigths of the particle according to the measurements
        weights = update(particles, weights, z, R=sensor_std_err, landmarks=landmarks)

        # If too few effective particles, resample
        if neff(weights) < N/2:
            indexes = systematic(weights)
            particles, weights = resampling_from_index(particles, weights, indexes)
        
        # Estimate the mean and variance of the particles according to their weights
        mu, var = estimate(particles, weights)
        history.append((particles.copy(), weights.copy(), mu, var))
    return history

ITER = 18
N = 5000

if __name__=="__main__":
    x_groundtruth = [[i, i] for i in range(ITER+1)]

    sensor_std_err = .1
    model_std_err = (.2, .05)

    u = np.tile([0, np.sqrt(2)], (ITER, 1))

    landmarks = np.array([
        [-1, 2],
        [5, 10],
        [12, 14],
        [18, 21]
    ])

    z = [(np.linalg.norm(landmarks - x_pos, axis=1) + (np.random.randn(len(landmarks)) * sensor_std_err)) for x_pos in x_groundtruth[1:]]

    np.random.seed(19)
    # history = particle_filter(N=N, controls=u, measurements=z, landmarks=landmarks, init_pos=((1,1, np.pi/4), (5, 5, np.pi/4)))
    history = particle_filter(N=N, controls=u, measurements=z, landmarks=landmarks)
    
    plt.figure()

    plt.scatter(history[0][0][:, 0], history[0][0][:, 1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    for i, state in enumerate(history[1:]):
        p0 = plt.scatter(state[0][:, 0], state[0][:, 1], color='k', marker=',', s=1)
        p1 = plt.scatter(x_groundtruth[i+1][0], x_groundtruth[i+1][1], marker='+', color='k', s=180, lw=3)
        p2 = plt.scatter(state[2][0], state[2][1], marker='s', color='r')
    
    plt.legend([p0, p1, p2], ['Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    print('final position error, variance:\n\t', state[2][:2] - x_groundtruth[-1], state[3][:2])
    plt.show()
