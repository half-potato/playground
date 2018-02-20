import numpy as np
import cv2
import matplotlib.pyplot as plt

class Kuramoto:
    def __init__(self, size, mean, std, coupling):
        """
        mean: float
          The mean frequency of oscillators in hertz
        """
        self.internal_freq = np.random.normal(mean, std, (1, size))
        self.phase = 2*np.pi*np.random.rand(1, size)
        self.size = size
        self.coupling = coupling
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        self.hist = self.internal_freq

    def update(self, dt):
        phase_rep = np.sin(np.repeat(self.phase, self.size, axis=0))
        phase_diff = np.sum(phase_rep - phase_rep.T, axis=0)
        dtheta = (self.internal_freq + self.coupling*phase_diff/self.size)
        self.phase = np.mod(self.phase + dtheta * dt, 2*np.pi)
        self.hist = np.vstack((self.hist, self.phase))

    def view(self):
        cv2.imshow("Display", cv2.normalize(self.phase, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    def view_hist(self):
        plt.plot(range(self.hist.shape[0]), np.sum(self.hist, axis=1))
        plt.show()
        #cv2.imshow("Display", cv2.normalize(self.hist, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    def view_fft(self):
        fft = np.fft.fft(self.phase).real
        cv2.imshow("Display", cv2.normalize(fft, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

if __name__ == "__main__":
    k = Kuramoto(30, 1, 0.01, 0.03)
    dt = .05
    while True:
        k.update(dt)
        k.view()
        #k.view_fft()
        if chr(cv2.waitKey(1) & 0xFF) == "q":
            break
    k.view_hist()
    cv2.waitKey(0)
