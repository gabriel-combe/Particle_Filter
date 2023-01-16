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
    for i in range(z.shape[1]):
        dist += (particles[:, :, i] - z[:, i])**2
    dist = np.sqrt(dist)
    print(dist)

    for i in range(particles.shape[1]):
        weights += ss.norm(0, np.sqrt(R[i])).pdf(dist[:, i])

    return weights


##################################
### SimplePosHeadingParticle2D ###

def SimplePosHeadingParticle2D_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    N = particles.shape[0]
    track_dim = particles.shape[1]

    particles[:, :, 2] += (u[:, 0] + (np.random.randn(N, track_dim) * Q_control[:, 0])) % 2*np.pi

    distance = (u[:, 1] * dt) + (np.random.randn(N, track_dim) * Q_control[:, 1])

    particles[:, :, 0] += np.cos(particles[:, :, 2]) * distance
    particles[:, :, 1] += np.sin(particles[:, :, 2]) * distance

    return particles

def SimplePosHeadingParticle2D_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    for i in range(particles.shape[1]):
        proba = 1.
        for k, landmark in enumerate(landmarks):
            distance = np.linalg.norm(particles[:, i, :2] - landmark, axis=1)
            proba *= ss.norm(distance, R[i]).pdf(z[i, k])
        weights *= proba

    return weights


##################################
###### ConstAccelParticle2D ######

def ConstAccelParticle2D_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    pass

def ConstAccelParticle2D_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    pass