import numpy as np
import cv2
import matplotlib.pyplot as plt

class Kuramoto2D:
    def __init__(self, shape, mean, std, coupling):
        """
        mean: float
          The mean frequency of oscillators in hertz
        """
        self.internal_freq = np.random.normal(mean, std, shape)
        self.phase = 2*np.pi*np.random.rand(*shape)
        self.shape = shape
        self.couplings = coupling
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        self.hist = self.internal_freq
        self.rs = []
        self.phis = []

    def update(self, dt):
        # Array is repeated along the first axis
        axis = 1
        tile_shape = (self.shape[axis],*np.repeat([1], len(self.shape)))
        phase_rep = np.tile(self.phase, tile_shape)
        phase_rep_swap = np.swapaxes(phase_rep, 0, 1+axis)
        phase_diff = self.couplings*np.sin(phase_rep_swap - phase_rep)
        phase_sum_1 = np.sum(phase_diff, axis=0)

        axis = 0
        tile_shape = (self.shape[axis],*np.repeat([1], len(self.shape)))
        print(tile_shape)
        phase_rep = np.tile(self.phase, tile_shape)
        print(phase_rep.shape)
        phase_rep_swap = np.swapaxes(phase_rep, 0, 1+axis)
        print(phase_rep_swap.shape)
        print(np.sin(phase_rep_swap - phase_rep).shape)
        print(self.couplings.shape)
        phase_diff = self.couplings*np.sin(phase_rep_swap - phase_rep)
        phase_sum = np.sum(phase_diff, axis=0)

        """
        # Repeat for second axis
        tile_shape = (np.max(self.shape),*np.repeat([1], len(self.shape)))
        #tile_shape = (30,*np.repeat([1], len(self.shape)))
        phase_rep = np.tile(self.phase, tile_shape)
        phase_rep_swap = np.swapaxes(phase_rep, 0, 1+np.argmax(self.shape))
        #phase_rep = np.swapaxes(phase_rep, 1, np.argmax(self.shape))
        phase_diff = self.couplings*np.sin(phase_rep - phase_rep_swap)
        phase_diff = self.couplings*np.sin(phase_rep_swap - phase_rep)
        phase_sum = np.sum(phase_diff, axis=0)
        """
        self.dtheta = (self.internal_freq + phase_sum/self.shape[0]/self.shape[1])
        self.phase = np.mod(self.phase + self.dtheta * dt, 2*np.pi)
        r, phi = self.order()
        self.rs.append(r)
        self.phis.append(phi)

    def order(self):
        o = np.mean(np.exp(self.phase * np.complex(0,1)))
        return abs(o), np.angle(o)

    def view(self):
        cv2.imshow("Display", cv2.normalize(self.phase, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    def view_hist(self):
        plt.subplot(2,1,1)
        plt.plot(range(len(self.rs)), self.rs)
        plt.subplot(2,1,2)
        plt.plot(range(len(self.phis)), self.phis)
        plt.show()

    def view_fft(self):
        fft = np.fft.fft(self.phase).real
        cv2.imshow("Display", cv2.normalize(fft, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

if __name__ == "__main__":
    shape = (20,30)
    k = Kuramoto2D(shape, 1, 0.03, 20*np.ones(shape))
    dt = .050
    while True:
        k.update(dt)
        k.view()
        print(k.order())
        #k.view_fft()
        if chr(cv2.waitKey(2) & 0xFF) == "q":
            break
    k.view_hist()
    cv2.waitKey(0)
