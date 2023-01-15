import numpy as np
import matplotlib.pyplot as plt
from ParticleFilter import ParticleFilter
from Particle import SimplePosHeadingParticle2D
from ResampleMethods import systematic_resample
from Models import SimplePosHeadingParticle2D_motion_model, SimplePosHeadingParticle2D_measurement_model

N = 5000
ITER = 18

def test():
    x_groundtruth = [[i, i] for i in range(ITER+1)]

    R = np.array([.1], dtype=float)
    Q_control = np.array([.2, .05], dtype=float)[np.newaxis, :]

    controls = np.tile([0, np.sqrt(2)], (ITER, 1))[:, np.newaxis, :]

    landmarks = np.array([
        [4, 1.5],
        [5, 10],
        [12, 14],
        [18, 16]
    ], dtype=float)

    measurements = np.array([(np.linalg.norm(landmarks - x_pos, axis=1) + (np.random.randn(len(landmarks)) * R[0])) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]

    pf = ParticleFilter(N, particle_struct=SimplePosHeadingParticle2D,
                        track_dim=1, control_dim=2,
                        ranges=((0, 20), (0, 20), (0, 2*np.pi)),
                        motion_model_fn=SimplePosHeadingParticle2D_motion_model,
                        measurement_model_fn=SimplePosHeadingParticle2D_measurement_model,
                        Q_control=Q_control,
                        R=R,
                        resample_method_fn=systematic_resample)
    
    plt.figure()

    plt.scatter(pf.particles[:, 0, 0], pf.particles[:, 0, 1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    for i, (u, z) in enumerate(zip(controls, measurements)):
        pf.predict(u, dt=1.)
        pf.update(z, args=([landmarks]))
        pf.resample(fraction=1./4.)
        pf.estimate()

        particles = plt.scatter(pf.particles[:, 0, 0], pf.particles[:, 0, 1], color='k', marker=',', s=1)
        x_true = plt.scatter(x_groundtruth[i+1][0], x_groundtruth[i+1][1], marker='+', color='k', s=180, lw=3)
        mean = plt.scatter(pf.mu[0, 0], pf.mu[0, 1], marker='s', color='r')

    l = plt.scatter(landmarks[:, 0], landmarks[:, 1], color='b', marker='^', s=180)

    plt.legend([l, particles, x_true, mean], ['Landmarks', 'Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    print('final position error, variance:\n\t', pf.mu[0, :2] - x_groundtruth[-1], pf.sigma[0, :2])
    plt.show()


if __name__=="__main__":
    test()