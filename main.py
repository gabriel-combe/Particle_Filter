import time
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from ParticleFilter import ParticleFilter
from Particle import SimplePosHeadingParticle2D, ConstAccelParticle2D
from ResampleMethods import systematic_resample

N = 5000
ITER = 50
DT = .01

def SimplePosHeadingParticle2D_test():
    x_groundtruth = np.array([[i*DT, i] for i in range(ITER+1)])

    R = np.array([[.1]])
    Q_control = np.array([.2, .05])[np.newaxis, :]

    controls = np.tile([0, np.sqrt(2)], (ITER, 1))[:, np.newaxis, :]
    
    init_pos = np.array([[[0, 0, 0],
                        [.4, .4, .2]]])

    landmarks = np.array([
        [4, 1.5],
        [5, 10],
        [12, 14],
        [18, 16]
    ])

    measurements = np.array([(np.linalg.norm(landmarks - x_pos, axis=1) + (np.random.randn(len(landmarks)) * R[0])) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]
    print(measurements[0])

    pf = ParticleFilter(N, particle_struct=SimplePosHeadingParticle2D,
                        track_dim=1, control_dim=2, # init_pos=init_pos,
                        ranges=np.array([[[0, 20], [0, 20], [0, 2*np.pi]]]),
                        Q_control=Q_control,
                        R=R,
                        resample_method_fn=systematic_resample)
    
    plt.figure()

    particles = [[deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 1])]]
    mean = [[0, 0]]

    for u, z in zip(controls, measurements):
        pf.forward(z, u, dt=DT, fraction=1./4., args=([landmarks]), verbose=2)

        particles.append([deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 1])])
        mean.append([pf.mu[0, 0], pf.mu[0, 1]])

    print('final position error, variance:\n\t', pf.mu[0, :2] - x_groundtruth[-1], pf.sigma[0, :2])

    p = plt.scatter(particles[0][0], particles[0][1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    for particle in particles[1:]:    
        p = plt.scatter(particle[0], particle[1], color='k', marker=',', s=1)
    
    x = plt.scatter(x_groundtruth[:, 0], x_groundtruth[:, 1], marker='+', color='cyan', s=180, lw=3)
    
    for avg in mean:
        m = plt.scatter(avg[0], avg[1], marker='s', color='r')

    l = plt.scatter(landmarks[:, 0], landmarks[:, 1], color='b', marker='^', s=180)
    
    plt.legend([l, p, x, m], ['Landmarks', 'Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    plt.show()


def ConstAccelParticle2DVel_test():
    # x_groundtruth = np.array([[i*DT, (i*DT)**2, 1, ((i*DT)**2 - ((i-1)*DT)**2)/DT] for i in range(int((ITER)/DT)+1)])
    # x_groundtruth = np.array([[i*DT, (i*DT)**3, 1, ((i*DT)**3 - ((i-1)*DT)**3)/DT] for i in range(int((ITER)/DT)+1)])
    x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT), 1, 10.*(np.sin(i*DT) - np.sin((i-1)*DT))/DT] for i in range(int((ITER)/DT)+1)])

    R = np.array([[.2, .2, .5, .5]])
    Q_motion = np.array([.3, .2, .7, .3, .2, .7])[np.newaxis, :]

    measurements = np.array([x_pos + (np.random.randn(4) * R[0]) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]

    init_pos = np.array([[[0, 1, 1, 0, 1, 2],
                        [.2, .3, .2, .2, .3, .2]]])

    pf = ParticleFilter(N, particle_struct=ConstAccelParticle2D,
                        track_dim=1, init_pos=init_pos,
                        # ranges=np.array([[0, 20], [-10, 10], [-5, 5], [0, 20], [-10, 10], [-5, 5]]),
                        Q_motion=Q_motion,
                        R=R,
                        resample_method_fn=systematic_resample)

    plt.figure()

    particles = [[deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])]]
    mean = [[init_pos[0, 0, 0], init_pos[0, 0, 3]]]
    
    mean_time = 0
    for i, z in enumerate(measurements):
        print(f'\nIteration {i+1}')

        start = time.time()
        pf.forward(z, dt=DT, fraction=1./4., verbose=2)
        mean_time += time.time() - start

        particles.append([deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])])
        mean.append([pf.mu[0, 0], pf.mu[0, 3]])

    mean_particle = np.mean(np.array(mean) - np.array(x_groundtruth)[:, :2], axis=0)
    print(f'\n\nTotal time for {measurements.shape[0]} -> {mean_time} seconds')
    print(f'\nAverage time for a particle filter pass: {mean_time / measurements.shape[0]} seconds')
    print('\nfinal position error, variance:\n\t', mean_particle, pf.sigma[0, ::3])

    p = plt.scatter(particles[0][0], particles[0][1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    # for particle in particles[1:]:    
    #     p = plt.scatter(particle[0], particle[1], color='k', marker=',', s=1)
    
    x = plt.scatter(x_groundtruth[:, 0], x_groundtruth[:, 1], marker='+', color='cyan', s=180, lw=3)
    
    for avg in mean:
        m = plt.scatter(avg[0], avg[1], marker='s', color='r')

    plt.legend([p, x, m], ['Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    plt.show()


def ConstAccelParticle2DVel2T_test():
    # x_groundtruth = np.array([[i*DT, (i*DT)**2, 1, ((i*DT)**2 - ((i-1)*DT)**2)/DT] for i in range(int((ITER)/DT)+1)])
    # x_groundtruth = np.array([[i*DT, (i*DT)**3, 1, ((i*DT)**3 - ((i-1)*DT)**3)/DT] for i in range(int((ITER)/DT)+1)])
    x_groundtruth = np.array([[i*DT, 10.*np.sin(i*DT), 1, 10.*(np.sin(i*DT) - np.sin((i-1)*DT))/DT] for i in range(int((ITER)/DT)+1)])

    R = np.array([[.2, .2, .5, .5], [.2, .2, .5, .5]])
    Q_motion = np.array([.3, .2, .7, .3, .2, .7])[np.newaxis, :]
    Q_motion = np.repeat(Q_motion, 2, axis=0)

    measurements = np.array([x_pos + (np.random.randn(4) * R[0]) for x_pos in x_groundtruth[1:]])[:, np.newaxis, :]
    measurements = np.repeat(measurements, 2, axis=1)

    init_pos = np.array([[[0, 1, 1, 0, 1, 2],
                        [.2, .3, .2, .2, .3, .2]],
                        [[0, 1, 1, 0, 1, 2],
                        [.2, .3, .2, .2, .3, .2]]])

    pf = ParticleFilter(N, particle_struct=ConstAccelParticle2D,
                        track_dim=2, init_pos=init_pos,
                        # ranges=np.array([[0, 20], [-10, 10], [-5, 5], [0, 20], [-10, 10], [-5, 5]]),
                        Q_motion=Q_motion,
                        R=R,
                        resample_method_fn=systematic_resample)

    plt.figure()

    particles = [[deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])]]
    mean = [[init_pos[0, 0, 0], init_pos[0, 0, 3]]]
    
    mean_time = 0
    for i, z in enumerate(measurements):
        print(f'\nIteration {i+1}')

        start = time.time()
        pf.forward(z, dt=DT, fraction=1./4., verbose=2)
        mean_time += time.time() - start

        particles.append([deepcopy(pf.particles[:, 0, 0]), deepcopy(pf.particles[:, 0, 3])])
        mean.append([pf.mu[0, 0], pf.mu[0, 3]])

    print(f'\n\nTotal time for {measurements.shape[0]} -> {mean_time} seconds')
    print(f'\nAverage time for a particle filter pass: {mean_time / measurements.shape[0]} seconds')
    print('\nfinal position error, variance:\n\t', pf.mu[0, ::3] - x_groundtruth[-1][:2], pf.sigma[0, ::3])

    p = plt.scatter(particles[0][0], particles[0][1], alpha=.20*np.sqrt(5000)/np.sqrt(N), color='g')

    # for particle in particles[1:]:    
    #     p = plt.scatter(particle[0], particle[1], color='k', marker=',', s=1)
    
    x = plt.scatter(x_groundtruth[:, 0], x_groundtruth[:, 1], marker='+', color='cyan', s=180, lw=3)
    
    for avg in mean:
        m = plt.scatter(avg[0], avg[1], marker='s', color='r')

    plt.legend([p, x, m], ['Particles', 'Actual', 'PF'], loc=4, numpoints=1)
    plt.show()

if __name__=="__main__":
    # SimplePosHeadingParticle2D_test()
    ConstAccelParticle2DVel_test()
    # ConstAccelParticle2DVel2T_test()