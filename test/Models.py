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
##### ConstAccelParticle2DVel ####

def ConstAccelParticle2DVel_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    N = particles.shape[0]
    track_dim = particles.shape[1]

    # X position
    particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt + np.random.randn(N, track_dim) * Q_model[:, 0]
    # X velocity
    particles[:, :, 1] += particles[:, :, 2] * dt + np.random.randn(N, track_dim) * Q_model[:, 1]
    # X acceleration
    particles[:, :, 2] += np.random.randn(N, track_dim) * Q_model[:, 2]
    
    # Y position
    particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt + np.random.randn(N, track_dim) * Q_model[:, 3]
    # Y velocity
    particles[:, :, 4] += particles[:, :, 5] * dt + np.random.randn(N, track_dim) * Q_model[:, 4]
    # Y acceleration
    particles[:, :, 5] += np.random.randn(N, track_dim) * Q_model[:, 5]

    return particles

def ConstAccelParticle2DVel_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray) -> np.ndarray:
    for i in range(particles.shape[1]):
        pos_error = np.sqrt(((particles[:, i, 0] - z[i, 0])/R[i][0])**2 + ((particles[:, i, 3] - z[i, 1])/R[i][1])**2)
        vel_error = np.sqrt(((particles[:, i, 1] - z[i, 2])/R[i][2])**2 + ((particles[:, i, 4] - z[i, 3])/R[i][3])**2)
        weights *= ss.norm(0, 1.).pdf(pos_error)
        weights *= ss.norm(0, 1.).pdf(vel_error)
    return weights


##################################
###### ConstAccelParticle2D ######

def ConstAccelParticle2D_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    N = particles.shape[0]
    track_dim = particles.shape[1]

    # X position
    particles[:, :, 0] += -.5 * particles[:, :, 2] * dt**2 + particles[:, :, 1] * dt + np.random.randn(N, track_dim) * Q_model[:, 0]
    # X velocity
    particles[:, :, 1] += particles[:, :, 2] * dt + np.random.randn(N, track_dim) * Q_model[:, 1]
    # X acceleration
    particles[:, :, 2] += np.random.randn(N, track_dim) * Q_model[:, 2]
    
    # Y position
    particles[:, :, 3] += -.5 * particles[:, :, 5] * dt**2 + particles[:, :, 4] * dt + np.random.randn(N, track_dim) * Q_model[:, 3]
    # Y velocity
    particles[:, :, 4] += particles[:, :, 5] * dt + np.random.randn(N, track_dim) * Q_model[:, 4]
    # Y acceleration
    particles[:, :, 5] += np.random.randn(N, track_dim) * Q_model[:, 5]

    return particles

def ConstAccelParticle2D_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray) -> np.ndarray:
    for i in range(particles.shape[1]):
        pos_error = np.sqrt((particles[:, i, 0] - z[i, 0])**2 + (particles[:, i, 3] - z[i, 1])**2)
        weights *= ss.norm(0, np.sqrt(R[i][0])).pdf(pos_error)
    return weights


##################################
###### ConstVelParticle2D ######

def ConstVelParticle2D_motion_model(particles: np.ndarray, Q_model: np.ndarray, dt: float, u: Optional[np.ndarray] =None, Q_control: Optional[np.ndarray] =None) -> np.ndarray:
    N = particles.shape[0]
    track_dim = particles.shape[1]

    # X position
    particles[:, :, 0] += particles[:, :, 1] * dt + np.random.randn(N, track_dim) * Q_model[:, 0]
    # X velocity
    particles[:, :, 1] += np.random.randn(N, track_dim) * Q_model[:, 1]
    
    # Y position
    particles[:, :, 3] += particles[:, :, 2] * dt + np.random.randn(N, track_dim) * Q_model[:, 2]
    # Y velocity
    particles[:, :, 4] += np.random.randn(N, track_dim) * Q_model[:, 3]

    return particles

def ConstVelParticle2D_measurement_model(particles: np.ndarray, weights: np.ndarray, z: np.ndarray, R: np.ndarray) -> np.ndarray:
    for i in range(particles.shape[1]):
        pos_error = np.sqrt((particles[:, i, 0] - z[i, 0])**2 + (particles[:, i, 3] - z[i, 1])**2)
        weights *= ss.norm(0, np.sqrt(R[i][0])).pdf(pos_error)
    return weights