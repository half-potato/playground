import numpy as np
import cupy as cp
import cv2, time
import matplotlib.pyplot as plt
import scipy.stats as st

class Kuramoto:
    def __init__(self, size, mean, std, coupling):
        """
        mean: float
          The mean frequency of oscillators in hertz
        """
        self.internal_freq = cp.random.normal(mean, std, (1, size))
        self.phase = 2*np.pi*cp.random.rand(1, size)
        self.size = size
        self.coupling = coupling

    def update(self, dt):
        phase_rep = cp.repeat(self.phase, self.size, axis=0)
        phase_diff = cp.sum(self.coupling*cp.sin(phase_rep.T - phase_rep), axis=0)
        dtheta = (self.internal_freq + phase_diff/self.size)
        self.phase = cp.mod(self.phase + dtheta * dt, 2*np.pi)
        self.hist = cp.vstack((self.hist, self.phase))
        r, phi = self.order()
        self.rs.append(r)
        self.phis.append(phi)

    def order(self):
        o = cp.mean(cp.exp(self.phase * np.complex(0,1)))
        return abs(o), cp.angle(o)

    def getPhaseIm(self, shape):
        return cp.asnumpy(cp.reshape(self.phase, shape))

def norm(img):
    return cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = cp.array(np.diff(st.norm.cdf(x)))
    kernel_raw = cp.sqrt(cp.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def shift(a, x, y):
    b = cp.roll(a, x, axis=1)
    if x >= 0:
        b[:, 0:x] = cp.zeros((a.shape[0], abs(x)))
    else:
        b[:, x:] = cp.zeros((a.shape[0], abs(x)))

    b = cp.roll(b, y, axis=0)
    if y >= 0:
        b[0:y, :] = cp.zeros((abs(y), a.shape[1]))
    else:
        b[y:, :] = cp.zeros((abs(y), a.shape[1]))
    return b

def ring_coupling(shape, weights):
    e = cp.eye(*shape)
    c = e.copy()
    for i in range(len(weights)-1):
        #c += weights[i]*cp.roll(e, i+1, axis=0)
        #c += weights[i]*cp.roll(e, i+1, axis=1)
        c += weights[i]*shift(e, i+1, 0)
        c += weights[i]*shift(e, 0, i+1)
    return c

def local_coupling(shape, kernel):
    axis_pad = ((0,shape[0]-kernel.shape[0]),(0, shape[1]-kernel.shape[1]))
    k_pad = cp.pad(kernel, axis_pad, "constant", constant_values=(0))
    coupling = cp.zeros((shape[0]**2, shape[1]**2))
    for i in range(shape[0]):
        for j in range(shape[1]):
            k = i*shape[0]+j
            coupling[k] = shift(k_pad, i-kernel.shape[1]//2, j-kernel.shape[0]//2)
    return coupling

# Tried params:
# size:40 coupling: 220 hz: 1 hzstd: 0.50 swirls
# size:60 coupling: 300 hz: 1 hzstd: 0.50 swirls
# size:120 coupling: 300 hz: 1 hzstd: 0.50 swirls
if __name__ == "__main__":
    size = 120
    #coupling = np.ones((size**2, size**2))
    #coupling = np.random.normal(1, .5, (size**2, size**2))
    #coupling = ring_coupling((size**2, size**2), [1, 1, 1, 0.75, .5, 0.25])
    coupling = cp.array(norm(cp.asnumpy(local_coupling((size, size), gkern(5, 2)))))
    #coupling = ring_coupling((size**2, size**2), [1, 1])
    k = Kuramoto(size**2, 1, 0.50, 1500*coupling)
    dt = .050
    while True:
        k.update(dt)
        k.view((size, size))
        k.store((size, size))
        #print(k.order())
        #k.view_fft()
        if chr(cv2.waitKey(1) & 0xFF) == "q":
            break
    k.play()
    k.save("kuramoto.avi")
    k.view_hist()
    cv2.waitKey(0)
