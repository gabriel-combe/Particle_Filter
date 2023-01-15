import numpy as np
import scipy.stats as ss
from typing import Optional

##################################
############ Default #############

def default_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    model = particles @ np.eye(particles.shape[2]) + np.random.randn(*particles.shape) * Q_model
    control = 0
    if u is not None:
        control = u @ np.eye(u.shape[2]) + np.random.randn(*u.shape) * Q_control
    return model + control

def default_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray) -> np.ndarray:
    dist = 0
    for i in range(z.shape[0]):
        dist += (particles[:, :, i] - z[:, i])**2
    dist = np.sqrt(dist)

    return weights + ss.multivariate_normal(np.zeros(particles.shape[1]), np.sqrt(R)).pdf(dist)

##################################
### SimplePosHeadingParticle2D ###

def SimplePosHeadingParticle2D_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    N = particles.shape[0]
    track_dim = particles.shape[1]

    particles[:, :, 2] += (u[:, 0] + (np.random.randn(N, track_dim) * Q_control[0])) % 2*np.pi

    distance = (u[:, 1] * dt) + (np.random.randn(N, track_dim) * Q_control[1])

    particles[:, :, 0] += np.cos(particles[:, :, 2]) * distance
    particles[:, :, 1] += np.sin(particles[:, :, 2]) * distance

    return particles

def SimplePosHeadingParticle2D_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, :, :2] - landmark, axis=2)
        print(particles[:, :, :2])
        print(np.linalg.norm(particles[:, :, :2] - landmark, axis=0))
        print(np.linalg.norm(particles[:, :, :2] - landmark, axis=1))
        print(np.linalg.norm(particles[:, :, :2] - landmark, axis=2))
        print(distance.squeeze(axis=1))

        weights *= ss.multivariate_normal(distance.squeeze(axis=1), R[0, 0]).pdf(z[i])
    
    return weights

particles = np.array([
    [[1, 2, 3], [1, 2, 3]],
    [[4, 5, 6], [1, 2, 3]],
    [[7, 8, 9], [1, 2, 3]],
    [[10, 11, 12], [1, 2, 3]]
], dtype=np.float64)

weights = np.random.random(4)

landmarks = np.array([
    [4, 1.5],
    [5, 10],
    [12, 14],
    [18, 16]
], dtype=np.float64)

z = np.array([[4.5, 5.5, 2.5, 6.], [5.5, 2.5, 6., 4.5]], dtype=np.float64)

u = np.array([
    [[1, 2], [1, 2]],
    [[4, 5], [1, 2]],
    [[7, 8], [1, 2]],
    [[10, 11], [1, 2]]
], dtype=np.float64)

Q_motion = np.array([np.ones(3)*0.2, np.ones(3)*0.5])
Q_control = np.array([np.ones(2)*0.5, np.ones(2)*0.1])
R = np.eye(1)*0.5

dt = 0.1

# print(particles)
# print(weights)
# print(u)
# print(z)
# print(Q_motion)
# print(Q_control)
# print(R)
# print(dt)
particles = SimplePosHeadingParticle2D_motion_model(particles, Q_motion, dt, u, Q_control)
print(particles[:, 0, :].shape)
weights = SimplePosHeadingParticle2D_measurement_model((particles[:, 0, :])[:, np.newaxis, :], weights, z[0], R, landmarks)
print(particles)
print(weights)
print(u)