import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from ParticleFilter import ParticleFilter
from Particle import SimplePosHeadingParticle2D, ConstAccelParticle2D
from ResampleMethods import systematic_resample
from Models import SimplePosHeadingParticle2D_motion_model, SimplePosHeadingParticle2D_measurement_model, ConstAccelParticle2DVel_motion_model, ConstAccelParticle2DVel_measurement_model, ConstAccelParticle2D_motion_model, ConstAccelParticle2D_measurement_model

N = 5000
ITER = 50
DT = .01

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

def ConstAccelParticle2DVel_test():
    # x_groundtruth = np.array([[i*DT, (i*DT)**2, 1, ((i*DT)**2 - ((i-1)*DT)**2)/DT] for i in range(int((ITER)/DT)+1)])
    # x_groundtruth = np.array([[i*DT, (i*DT)**3, 1, ((i*DT)**3 - ((i-1)*DT)**3)/DT] for i in range(int((ITER)/DT)+1)])
    # x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT), DT, DT*10.*np.cos(i*DT)] for i in range(int((ITER)/DT)+1)])
    x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT), 1, 10.*(np.sin(i*DT) - np.sin((i-1)*DT))/DT] for i in range(int((ITER)/DT)+1)])

    R = np.array([[.2, .2, .5, .5]], dtype=float)
    Q_motion = np.array([.3, .2, .7, .3, .2, .7], dtype=float)[np.newaxis, :]

    measurements = np.array([x_pos + (np.random.randn(4) * R[0]) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]

    init_pos = [[0, 1, 1, 0, 1, 2],
                [.2, .3, .2, .2, .3, .2]]

    pf = ParticleFilter(N, particle_struct=ConstAccelParticle2D,
                        track_dim=1, init_pos=init_pos,
                        # ranges=np.array([[0, 20], [-10, 10], [-5, 5], [0, 20], [-10, 10], [-5, 5]]),
                        motion_model_fn=ConstAccelParticle2DVel_motion_model,
                        measurement_model_fn=ConstAccelParticle2DVel_measurement_model,
                        Q_motion=Q_motion,
                        R=R,
                        resample_method_fn=systematic_resample)

    plt.figure()

    particles = [[deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])]]
    mean = [[init_pos[0, 0], init_pos[0, 3]]]
    
    import time
    mean_time = 0
    for i, z in enumerate(measurements):
        print(f'\nIteration {i+1}')

        start = time.time()
        pf.forward(z, dt=DT, fraction=1./4., verbose=2)
        mean_time += time.time() - start

        particles.append([deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])])
        mean.append([pf.mu[0, 0], pf.mu[0, 3]])

    mean_time /= measurements.shape[0]

    print(mean_time)
    print(x_groundtruth.shape[0]-1)

    p = plt.scatter(particles[0][0], particles[0][1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    # for particle in particles[1:]:    
    #     p = plt.scatter(particle[0], particle[1], color='k', marker=',', s=1)
    
    x = plt.scatter(x_groundtruth[:, 0], x_groundtruth[:, 1], marker='+', color='cyan', s=180, lw=3)
    
    for avg in mean:
        m = plt.scatter(avg[0], avg[1], marker='s', color='r')

    plt.legend([p, x, m], ['Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    # plt.xlim((0, 20))
    # plt.ylim((0, 20))
    print('final position error, variance:\n\t', pf.mu[0, ::3] - x_groundtruth[-1][:2], pf.sigma[0, ::3])
    plt.show()


def ConstAccelParticle2D_test():
    # x_groundtruth = np.array([[i*DT, (i*DT)**2] for i in range(int((ITER)/DT)+1)])
    x_groundtruth = np.array([[i*DT, (i*DT - 1)**3] for i in range(int((ITER)/DT)+1)])
    # x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT)] for i in range(int((ITER)/DT)+1)])
    # x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT)] for i in range(int((ITER)/DT)+1)])

    R = np.array([[.2, .2]], dtype=float)
    Q_motion = np.array([.3, .2, .7, .3, .2, .7], dtype=float)[np.newaxis, :]

    measurements = np.array([x_pos + (np.random.randn(R[0].shape) * R[0]) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]

    init_pos = np.array([[0, 1, 1, 0, 1, 2],[.2, .3, .2, .2, .3, .2]])

    pf = ParticleFilter(N, particle_struct=ConstAccelParticle2D,
                        track_dim=1, init_pos=init_pos,
                        # ranges=np.array([[0, 20], [-10, 10], [-5, 5], [0, 20], [-10, 10], [-5, 5]]),
                        motion_model_fn=ConstAccelParticle2D_motion_model,
                        measurement_model_fn=ConstAccelParticle2D_measurement_model,
                        Q_motion=Q_motion,
                        R=R,
                        resample_method_fn=systematic_resample)

    plt.figure()

    particles = [[deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])]]
    x_true = [[x_groundtruth[0][0], x_groundtruth[0][1]]]
    mean = [[init_pos[0, 0], init_pos[0, 3]]]
    
    import time
    mean_time = 0
    for i, z in enumerate(measurements):
        start = time.time()
        pf.predict(dt=DT)
        pf.update(z)
        pf.resample(fraction=1./4.)
        pf.estimate()
        mean_time += time.time() - start

        particles.append([deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])])
        x_true.append([x_groundtruth[i+1, 0], x_groundtruth[i+1, 1]])
        mean.append([pf.mu[0, 0], pf.mu[0, 3]])
        print(f'Iteration {i+1}')
        print(f'Mean:\n\tposition {pf.mu[0, ::3]}\n\tvelocity {pf.mu[0, 1::3]}\n\tacceleration {pf.mu[0, 2::3]}')
        print(f'STD:\n\tposition {pf.sigma[0, ::3]}\n\tvelocity {pf.sigma[0, 1::3]}\n\tacceleration {pf.sigma[0, 2::3]}\n\n')

    mean_time /= measurements.shape[0]

    print(mean_time)
    print(x_groundtruth.shape[0]-1)

    p = plt.scatter(particles[0][0], particles[0][1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    # for particle in particles[1:]:    
    #     p = plt.scatter(particle[0], particle[1], color='k', marker=',', s=1)
    
    for x_ground in x_true:
        x = plt.scatter(x_ground[0], x_ground[1], marker='+', color='cyan', s=180, lw=3)
    
    for avg in mean:
        m = plt.scatter(avg[0], avg[1], marker='s', color='r')

    plt.legend([p, x, m], ['Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    # plt.xlim((0, 20))
    # plt.ylim((0, 20))
    print('final position error, variance:\n\t', pf.mu[0, ::3] - x_groundtruth[-1][:2], pf.sigma[0, ::3])
    plt.show()

if __name__=="__main__":
    # SimplePosHeadingParticle2D_test()
    ConstAccelParticle2DVel_test()