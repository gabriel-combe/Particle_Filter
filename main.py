import numpy as np
import matplotlib.pyplot as plt
from ParticleFilter import ParticleFilter
from Particle import SimplePosHeadingParticle2D, ConstAccelParticle2D
from ResampleMethods import systematic_resample
from Models import SimplePosHeadingParticle2D_motion_model, SimplePosHeadingParticle2D_measurement_model, ConstAccelParticle2D_motion_model, ConstAccelParticle2D_measurement_model

N = 5000
ITER = 20
DT = .001

def SimplePosHeadingParticle2D_test():
    x_groundtruth = [[i*DT, i] for i in range(ITER+1)]

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
        pf.predict(u, dt=DT)
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

def ConstAccelParticle2D_test():
    # x_groundtruth = [[i*DT, (i*DT)**2, DT, 2*(i*DT)] for i in range(ITER+1)]
    # x_groundtruth = [[i*DT, (i*DT)**3, DT, 3*(i*DT)**2] for i in range(ITER+1)]
    x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT), DT, DT*10.*np.cos(i*DT)] for i in range(int((ITER)/DT)+1)])

    R = np.array([[.2, .2, .5, .5]], dtype=float)
    Q_motion = np.array([.3, .2, .7, .3, .2, .7], dtype=float)[np.newaxis, :]

    measurements = np.array([x_pos + (np.random.randn(4) * R[0]) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]

    init_pos = np.array([[0, 1, 1, 0, 1, 2],[.01, .3, .4, .01, .1, .05]])

    pf = ParticleFilter(N, particle_struct=ConstAccelParticle2D,
                        track_dim=1, init_pos=init_pos,
                        # ranges=np.array([[0, 20], [-10, 10], [-5, 5], [0, 20], [-10, 10], [-5, 5]]),
                        motion_model_fn=ConstAccelParticle2D_motion_model,
                        measurement_model_fn=ConstAccelParticle2D_measurement_model,
                        Q_motion=Q_motion,
                        R=R,
                        resample_method_fn=systematic_resample)

    plt.figure()

    plt.scatter(pf.particles[:, 0, 0], pf.particles[:, 0, 3], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')
    import time
    mean_time = 0
    for i, z in enumerate(measurements):
        start = time.time()
        pf.predict(dt=DT)
        pf.update(z)
        pf.resample(fraction=1./4.)
        pf.estimate()
        mean_time += time.time() - start

        particles = plt.scatter(pf.particles[:, 0, 0], pf.particles[:, 0, 3], color='k', marker=',', s=1)
        x_true = plt.scatter(x_groundtruth[i+1][0], x_groundtruth[i+1][1], marker='+', color='k', s=180, lw=3)
        mean = plt.scatter(pf.mu[0, 0], pf.mu[0, 3], marker='s', color='r')
        print(f'Mean:\n\tposition {pf.mu[0, ::3]}\n\tvelocity {pf.mu[0, 1::3]}\n\tacceleration {pf.mu[0, 2::3]}')
        print(f'STD:\n\tposition {pf.sigma[0, ::3]}\n\tvelocity {pf.sigma[0, 1::3]}\n\tacceleration {pf.sigma[0, 2::3]}\n\n')

    mean_time /= measurements.shape[0]

    print(mean_time)
    print(x_groundtruth.shape[0])
    plt.legend([particles, x_true, mean], ['Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim((0, 20))
    plt.ylim((0, 20))
    print('final position error, variance:\n\t', pf.mu[0, ::3] - x_groundtruth[-1][:2], pf.sigma[0, ::3])
    plt.show()


if __name__=="__main__":
    # SimplePosHeadingParticle2D_test()
    ConstAccelParticle2D_test()