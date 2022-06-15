from jax import grad
import numpy as np
import jax.numpy as jnp

from solver.LalanneClass import LalanneBase

# TEST git
class RCWAOptimizer:

    def __init__(self, gt, model):
        self.gt = gt
        self.model = model
        pass

    def get_difference(self):
        spectrum_gt = np.hstack(self.gt.spectrum_R, self.gt.spectrum_T)
        spectrum_model = np.hstack(self.model.spectrum_R, self.model.spectrum_T)
        residue = spectrum_model - spectrum_gt
        loss = np.linalg.norm(residue)

class TT:
    def __init__(self):
        pass


if __name__ == '__main__':
    grating_type = 0
    aa = TT()
    print(1)
    gt = LalanneBase(grating_type,wls=np.linspace(0.5, 2.3, 40), fourier_order=1,)
    # model = LalanneBase(grating_type, wls=np.linspace(0.5, 2.3, 40), fourier_order=1, thickness=[x])


    def loss(x):
        print(2)
        model = LalanneBase(grating_type, wls=np.linspace(0.5, 2.3, 40), fourier_order=1, thickness=[x])
        print(3)
        model.lalanne_1d()

        gap = jnp.linalg.norm(model.spectrum_r - gt.spectrum_r)
        return gap

    grad_loss = grad(loss)
    print(grad_loss(1.0))


    pass
