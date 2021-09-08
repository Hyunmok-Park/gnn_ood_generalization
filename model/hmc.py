import numpy as np
import networkx as nx
EPS = float(np.finfo(np.float32).eps)
from tqdm import tqdm

class HMC(object):

    def __init__(self, A, J, b, seed=1234):
        """
          Belief Propagation for Binary Markov Models

          A: shape N X N, binary adjacency matrix
          J: shape N X N, parameters of pairwise term
          b: shape N X 1, parameters of unary term
        """
        super(HMC, self).__init__()
        self.A = A
        self.J = np.multiply(J, A)
        self.b = b
        self.num_nodes = A.shape[0]
        self.num_states = 2
        self.states = [1.0, -1.0]
        self.seed = seed
        self.npr = np.random.RandomState(seed=seed)

    def inference(self, travel_time=12.5, num_sample=10000, burn_in=1000):

        y_init = abs(self.npr.normal(0, 1, size=self.num_nodes))
        y_last = y_init
        Ys = np.zeros((burn_in + num_sample, self.num_nodes))
        Ys[0,:] = y_init

        # wall_hits = 0
        # wall_crosses = 0
        print("HMC")
        for i in tqdm(range(1, burn_in + num_sample)):
            stop, j = False, -1

            # initial velocity/momentum q
            q = self.npr.normal(0, 1, size=self.num_nodes)
            # auxiliary variable y
            y = y_last

            # records how much time the particle already moved
            tt = 0
            s = np.sign(y)

            while True:
                q0 = q
                y0 = y
                phi = np.arctan2(y0, q0)  # -pi < phi < +pi

                # find the first time constraint becomes zero
                wt1 = -phi  # time at which coordinates hit the walls
                wt1[np.where(phi > 0)] = np.pi - phi[np.where(phi > 0)]

                # -------------------------------------------------------------
                # if there was a previous reflection (j>0)
                # and there is a potential reflection at the sample plane
                # make sure that a new reflection at j is not found because of numerical error
                if j >= 0:
                    tt1 = wt1[j]
                    if abs(tt1) < EPS:
                        wt1[j] = np.inf

                min_wt, j = np.min(wt1), np.argmin(wt1)
                tt = tt + min_wt

                if tt >= travel_time:
                    min_wt = min_wt - (tt - travel_time)
                    stop = True
                #else:
                #    wall_hits = wall_hits + 1

                # --------------------------------------------------------------
                # move the particle a time min_wt
                y = q0 * np.sin(min_wt) + y0 * np.cos(min_wt)
                q = q0 * np.cos(min_wt) - y0 * np.sin(min_wt)

                if stop:
                    break

                y[j] = 0  # Wall hit!

                # Eq (9): q^2(+) = q^2(-) + 2*delta
                delta = 2 * (self.J[j,:].dot(s) + self.b[j])
                q_new = q[j] ** 2 + 2 * np.sign(q[j]) * delta
                if q_new > 0:
                    q[j] = np.sqrt(q_new) * np.sign(q[j])
                    s[j] = -s[j]
                    #wall_crosses = wall_crosses + 1
                else:
                    q[j] = -q[j]

            Ys[i,:] = y
            y_last = y

        Ys = Ys[burn_in:]
        #prob = np.sum((np.sign(Ys)+1)/2, axis=0, keepdims=False) / num_sample

        prob_log = []
        for i in range(num_sample):
            prob = np.sum((np.sign(Ys[:i + 1, :]) + 1) / 2, axis=0, keepdims=False) / (i + 1)
            prob_log += [list(prob)]

        return np.stack([prob, 1 - prob], axis=0).transpose()#, np.array(prob_log)


if __name__ == '__main__':
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    J = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, -0.8, 0.9]])
    J = (J + J.transpose()) / 2.0
    # J = np.ones((3, 3)) * 0.5
    b = np.ones((3, 1)) * 0.5
    T = 12.5
    ns = 5000
    burnIn = 1000

    model = HMC(A, J, b)
    P = model.inference(travel_time=T, num_sample=ns, burn_in=burnIn)
    print(P)
