CUDA = False
import numpy as np
if CUDA:
    import cupy as cp
    import KuramotoCuda as kc
else:
    import Kuramoto as kc
import cv2, time
import matplotlib.pyplot as plt
import scipy.stats as st

class KuramotoLogger:
    def __init__(self, size, mean, std, coupling, keep_hist=False):
        """
        mean: float
          The mean frequency of oscillators in hertz
        """
        self.kura = kc.Kuramoto(size, mean, std, coupling)
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        self.rs = []
        self.phis = []
        self.ims = []
        self.keep_hist = keep_hist
        if keep_hist:
            self.hist = self.kura.getPhaseIm((1,size))

    def update(self, dt):
        self.kura.update(dt)
        if self.keep_hist:
            self.hist = np.vstack((self.hist, self.kura.getPhaseIm(1, self.kura.size)))
        r, phi = self.kura.order()
        self.rs.append(r)
        self.phis.append(phi)

    def view(self, shape):
        im = self.kura.getPhaseIm(shape)
        norm_im = cv2.normalize(im, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        col_im = cv2.applyColorMap(np.uint8(norm_im), cv2.COLORMAP_PARULA)
        cv2.imshow("Display", col_im)

    def store(self, shape):
        im = self.kura.getPhaseIm(shape)
        norm_im = cv2.normalize(im, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        col_im = cv2.applyColorMap(np.uint8(norm_im), cv2.COLORMAP_PARULA)
        self.ims.append(col_im)

    def save(self, path):
        shape = self.ims[0].shape
        shape = (shape[0], shape[1])
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('H','F','Y','U'), 15, shape, 1)
        for i in self.ims:
            out.write(i)
        out.release()

    def save_upscale(self, path, pixelwise, uppyr):
        shape = self.ims[0].shape
        final_shape = (shape[0]*pixelwise*2**upyr, shape[1]*pixelwise*2**upyr)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('H','F','Y','U'), 15, final_shape, 1)
        for i in self.ims:
            f_n = cv2.resize(frame, (shape[0]*pixelwise, shape[1]*pixelwise), interpolation=cv2.INTER_NEAREST)
            for j in range(uppyr):
                i = cv2.pyrUp(i)
            out.write(i)
        out.release()

    def play(self):
        for i in self.ims:
            cv2.imshow("Display", i)
            if chr(cv2.waitKey(2) & 0xFF) == "q":
                break

    def view_hist(self):
        plt.subplot(3,1,1)
        plt.plot(range(self.hist.shape[0]), np.sum(self.hist, axis=1))
        plt.subplot(3,1,2)
        plt.plot(range(len(self.rs)), self.rs)
        plt.subplot(3,1,3)
        plt.plot(range(len(self.phis)), self.phis)
        plt.show()
        cv2.imshow("Display", norm(self.hist))

def norm(img):
    return cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Tried params:
# size:40 coupling: 220 hz: 1 hzstd: 0.50 swirls
# size:60 coupling: 300 hz: 1 hzstd: 0.50 swirls
# size:120 coupling: 1500 hz: 1 hzstd: 0.50 swirls
if __name__ == "__main__":
    size = 50
    #coupling = np.ones((size**2, size**2))
    if CUDA:
        coupling = cp.array(norm(cp.asnumpy(kc.local_coupling(
            (size, size), kc.gkern(5, 2)))))
    else:
        coupling = kc.local_coupling((size, size), kc.gkern(5, 2))
    k = KuramotoLogger(size**2, 1, 0.50, 1500*coupling)
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
    #k.save("kuramoto_%i.hfyu" % size)
    k.save_upscale("kuramoto_%i.hfyu" % size, 4, 2)
    k.view_hist()
    cv2.waitKey(0)
