from autograd import grad
import numpy as np
import autograd.numpy as anp

from solver.LalanneClass import LalanneBase


class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass

    def get_difference(self):
        spectrum_gt = anp.hstack(self.gt.spectrum_R, self.gt.spectrum_T)
        spectrum_model = anp.hstack(self.model.spectrum_R, self.model.spectrum_T)
        residue = spectrum_model - spectrum_gt
        loss = anp.linalg.norm(residue)


if __name__ == '__main__':
    grating_type = 0
    gt = LalanneBase(grating_type, wls=anp.linspace(0.5, 2.3, 40), fourier_order=8)
    gt.lalanne_1d()

    def loss(x):

        model = LalanneBase(grating_type, wls=anp.linspace(0.5, 2.3, 40), fourier_order=8, thickness=[x])

        model.lalanne_1d()

        # calculate norm
        aa = np.asarray(model.spectrum_r) - np.asarray(gt.spectrum_r)
        gap = (np.sum(aa ** 2) / float(len(aa))) ** 0.5

        return gap



    pass
