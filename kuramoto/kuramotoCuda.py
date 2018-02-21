import numpy as np
import cupy as cp
import cv2, time
import matplotlib.pyplot as plt
import scipy.stats as st

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

class Kuramoto:
    def __init__(self, size, mean, std, coupling):
        """
        mean: float
          The mean frequency of oscillators in hertz
        """
        self.internal_freq = cp.random.normal(mean, std, (1, size))
        self.phase = 2*cp.pi*cp.random.rand(1, size)
        self.size = size
        self.coupling = coupling
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        self.hist = self.internal_freq
        self.rs = []
        self.phis = []

    def update(self, dt):
        phase_rep = cp.repeat(self.phase, self.size, axis=0)
        phase_diff = cp.sum(self.coupling*cp.sin(phase_rep.T - phase_rep), axis=0)
        dtheta = (self.internal_freq + phase_diff/self.size)
        self.phase = cp.mod(self.phase + dtheta * dt, 2*cp.pi)
        self.hist = cp.vstack((self.hist, self.phase))
        r, phi = self.order()
        self.rs.append(r)
        self.phis.append(phi)

    def order(self):
        o = cp.mean(cp.exp(self.phase * cp.complex(0,1)))
        return abs(o), cp.angle(o)

    def view(self, shape):
        cv2.imshow("Display", norm(cp.reshape(self.phase, shape)))

    def view_hist(self):
        plt.subplot(3,1,1)
        plt.plot(range(self.hist.shape[0]), cp.sum(self.hist, axis=1))
        plt.subplot(3,1,2)
        plt.plot(range(len(self.rs)), self.rs)
        plt.subplot(3,1,3)
        plt.plot(range(len(self.phis)), self.phis)
        plt.show()
        cv2.imshow("Display", norm(self.hist))

    def view_fft(self):
        fft = cp.fft.fft(self.phase).real
        cv2.imshow("Display", norm(fft))

"""
def shift(a, num, axis):
    if num == 0:
        return a
    b = np.roll(a, num, axis=axis)
    s = np.arange(-num) if (num < 0) else a.shape[axis]-np.arange(num)-1
    ax = (axis+1) % len(a.shape)
    ones = np.zeros_like(np.take(b, s, axis=ax))
    b = np.delete(b, s, axis=ax)
    return np.concatenate((b, ones), axis=ax)
"""

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
    coupling = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            k = i*shape[0]+j
            print(i, j)
            k_shifted = shift(k_pad, i-kernel.shape[1]//2, j-kernel.shape[0]//2)
            coupling.append(k_shifted.flatten())
    return cp.array(coupling)

if __name__ == "__main__":
    size = 40
    #coupling = np.ones((size**2, size**2))
    #coupling = np.random.normal(1, .5, (size**2, size**2))
    #coupling = ring_coupling((size**2, size**2), [1, 1, 1, 0.75, .5, 0.25])
    coupling = local_coupling((size, size), gkern(5, 2))
    #coupling = cp.array(coupling.tolist())
    print(coupling)
    #coupling = ring_coupling((size**2, size**2), [1, 1])
    k = Kuramoto(size**2, 1, 0.50, 150*coupling)
    dt = .050
    while True:
        k.update(dt)
        k.view((size, size))
        #print(k.order())
        #k.view_fft()
        if chr(cv2.waitKey(1) & 0xFF) == "q":
            break
    k.view_hist()
    cv2.waitKey(0)
