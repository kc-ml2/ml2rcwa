# import numpy as np
from scipy.linalg import circulant
import jax.numpy as np

def to_conv_mat(permittivities, fourier_order):
    # FFT scaling
    # https://kr.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft#:~:text=the%20matlab%20fft%20outputs%202,point%20is%20the%20parseval%20equation.
    ff = 2 * fourier_order + 1

    # TODO: check whether 1D case is correct or not. I think I actually didn't test it at all.
    if len(permittivities[0].shape) == 1:  # 1D
        res = np.zeros((len(permittivities), 2*fourier_order+1, 2*fourier_order+1)).astype('complex')

        for i, pmtvy in enumerate(permittivities):
            pmtvy_fft = np.fft.fftn(pmtvy / pmtvy.size)
            pmtvy_fft = np.fft.fftshift(pmtvy_fft)

            center = len(pmtvy_fft) // 2
            pmtvy_fft_cut = (pmtvy_fft[-ff + center: center+ff+1])
            A = np.roll(circulant(pmtvy_fft_cut.flatten()), (pmtvy_fft_cut.size + 1) // 2, 0)
            res = res.at[i].set(A[:2*fourier_order+1, :2*fourier_order+1])
            # res[i] = circulant(pmtvy_fft_cut)

    else:  # 2D
        res = np.zeros((len(permittivities), ff ** 2, ff ** 2)).astype('complex')

        for i, pmtvy in enumerate(permittivities):
            pmtvy_fft = np.fft.fftn(pmtvy / pmtvy.size)
            pmtvy_fft = np.fft.fftshift(pmtvy_fft)

            # From Zhaonat.
            # https://math.stackexchange.com/questions/30245/are-fourier-coefficients-always-symmetric
            # TODO: Can this be improved?
            p0, q0 = np.array(pmtvy_fft.shape) // 2

            Af = pmtvy_fft.T
            P=Q=fourier_order
            p = list(range(-P, P + 1))  # array of size 2Q+1
            q = list(range(-Q, Q + 1))

            C = np.zeros(((2*P+1)**2, (2*P+1)**2))
            C = C.astype(complex)
            for qrow in range(2 * Q + 1):  # remember indices in the arrary are only POSITIVE
                for prow in range(2 * P + 1):  # outer sum
                    # first term locates z plane, 2nd locates y column, prow locates x
                    row = (qrow) * (2 * P + 1) + prow  # natural indexing
                    for qcol in range(2 * Q + 1):  # inner sum
                        for pcol in range(2 * P + 1):
                            col = (qcol) * (2 * P + 1) + pcol  # natural indexing
                            pfft = p[prow] - p[pcol]  # get index in Af; #index may be negative.
                            qfft = q[qrow] - q[qcol]
                            C[row, col] = Af[q0 + pfft, p0 + qfft]  # index may be negative.
            res[i] = C

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(abs(res[0]), cmap='jet')
    # plt.colorbar()
    # plt.show()

    return res


def draw_1d(patterns, resolution=1001):
    # fill_factor is not exactly implemented.

    res = np.zeros((len(patterns), resolution))

    for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns):

        permittivity = np.ones(resolution)
        cut = int(resolution * fill_factor)

        # permittivity[:cut] *= n_ridge ** 2
        # permittivity[cut:] *= n_groove ** 2
        # res[i] = permittivity

        permittivity = permittivity.at[:cut].set(n_ridge ** 2)
        permittivity = permittivity.at[cut:].set(n_groove ** 2)
        res = res.at[i].set(permittivity)

    return res


def draw_2d(patterns, resolution=1001):

    # TODO: Implement
    res = np.ndarray((len(patterns), resolution, resolution))

    for i, (n_ridge, n_groove, fill_factor) in enumerate(patterns):

        permittivity = np.ones((resolution, resolution))
        cut = int(resolution * fill_factor)
        permittivity[:, :] *= n_groove ** 2
        permittivity[:cut*2, :cut] *= n_ridge ** 2
        res[i] = permittivity

    return res
