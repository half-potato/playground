import numpy as np
import cv2

def get(arr, *args):
    v = []
    for i in arr:
        v.append(i[tuple(args)])
    return np.array(v)

class Swarm:
    def __init__(self, particle_num, domain, min_fn, relations):
        """
        domain: array of tuples of floats
            [(min, max), (min, max), ...]
        function: fn: [float]->float
            some vectorized function of the matrix rowwise, ie x[0] + x[1]**2
        """
        self.particle_num = particle_num
        self.domain = domain
        self.min_fn = min_fn
        self.positions = np.array([i[0] + (i[1]-i[0])*np.random.rand(particle_num) for i in domain])
        self.velocities = np.array([-abs(i[1]-i[0]) + 2*abs(i[1]-i[0])*np.random.rand(particle_num) for i in domain])
        self.p_best = self.positions.copy()
        self.p_best_val = min_fn(self.positions)
        self.g_best_val = np.min(self.p_best_val)
        i = np.unravel_index(np.argmax(-self.p_best_val), self.p_best_val.shape)
        self.g_best = get(self.positions, *i)

    def step(self, inertia, p_draw, g_draw):
        # Update p_best
        out = self.min_fn(self.positions)
        p_best_mask = np.less(out, self.p_best_val)
        p_best_mask_rep = np.repeat(np.expand_dims(p_best_mask, 0), len(self.domain), axis=0)
        self.p_best_val[p_best_mask] = out[p_best_mask]
        self.p_best[p_best_mask_rep] = self.positions[p_best_mask_rep]
        # Update g_best
        g_best = np.min(out)
        if g_best < self.g_best_val:
            i = np.unravel_index(np.argmax(-out), out.shape)
            self.g_best = get(self.positions, *i)
            self.g_best_val = g_best
        # Update velocities
        weighted_p_best = (self.p_best-self.positions)*np.random.rand(len(self.domain), self.particle_num)
        weighted_g_best = np.array(
            [self.g_best[i]-self.positions[i] * np.random.rand(self.particle_num) for i in range(len(self.domain))])
        self.velocities = inertia*self.velocities + p_draw*weighted_p_best + g_draw*weighted_g_best
        # Update positions
        # Add bounds checking
        self.positions += self.velocities

if __name__ == "__main__":
    def fn(x):
        return np.sin(x[0])**2 + np.cos(x[1])**2
    s = Swarm(500, ((-5, 5), (-5, 5)), fn, None)
    for i in range(100):
        s.step(0.5, 0.3, 0.3)
        print(s.g_best_val)
