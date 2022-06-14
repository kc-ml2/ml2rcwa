import copy

import matplotlib.pyplot as plt
from solver.convolution_matrix import *

import jax.numpy as np

class LalanneBase:
    def __init__(self, grating_type, n_I=1, n_II=1, theta=0, phi=0, psi=0, fourier_order=1,
                 period=0.7, wls=np.linspace(0.5, 2.3, 400), polarization=0,
                 patterns=None, thickness=None):

        self.grating_type = grating_type  # TE, TM, Conical
        self.n_I = n_I
        self.n_II = n_II

        self.theta = theta * np.pi / 180
        self.phi = phi * np.pi / 180
        self.psi = psi * np.pi / 180

        self.fourier_order = fourier_order
        self.ff = 2 * self.fourier_order + 1
        self.period = period

        self.wls = wls

        self.polarization = polarization  # TE 0, TM 1

        # permittivity in grating layer
        # self.patterns = [[3.48, 1, 0.3]] if patterns is None else patterns
        self.patterns = [['SILICON', 1, 0.3]] if patterns is None else patterns
        self.thickness = [0.46] if thickness is None else thickness

        self.spectrum_r = np.zeros(len(wls))
        self.spectrum_t = np.zeros(len(wls))

    def permittivity_mapping(self, wl, oneover=False):

        nk_pool = {'SILICON': np.array([[300, 400, 500, 600, 700], [3.1, 3.4, 5, 6.4, 2]])}
        nk_pool = {'SILICON': np.array([[100, 400, 500, 600, 900], [3.48, 3.48, 3.48, 3.48, 3.48]])}

        patterns = copy.deepcopy(self.patterns)

        for i, layer in enumerate(self.patterns):
            nk_table_of_interest = nk_pool[layer[0].upper()]
            patterns[i][0] = np.interp(wl, *nk_table_of_interest)

        if oneover:
            patterns = np.array(patterns)
            patterns[:, 0] = 1 / patterns[:, 0]

        if type(self.period) in [float, int] or len(self.period) == 1:
            pmtvy = draw_1d(patterns)
        else:
            pmtvy = draw_2d(patterns)
        conv_all = to_conv_mat(pmtvy, self.fourier_order)

        return conv_all

    def lalanne_1d(self):

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(2 * self.fourier_order + 1)
        # delta_i0[self.fourier_order] = 1
        delta_i0 = delta_i0.at[self.fourier_order].set(1)

        for i, wl in enumerate(self.wls):
            k0 = 2 * np.pi / wl

            E_conv_all = self.permittivity_mapping(wl)

            kx_vector = k0 * (self.n_I * np.sin(self.theta) - fourier_indices * (wl / self.period)).astype('complex')

            k_I_z = (k0 ** 2 * self.n_I ** 2 - kx_vector ** 2) ** 0.5
            k_II_z = (k0 ** 2 * self.n_II ** 2 - kx_vector ** 2) ** 0.5

            k_I_z = k_I_z.conjugate()
            k_II_z = k_II_z.conjugate()

            Kx = np.diag(kx_vector / k0)

            f = np.eye(2 * self.fourier_order + 1)

            if self.polarization == 0:  # TE
                Y_I = np.diag(k_I_z / k0)
                Y_II = np.diag(k_II_z / k0)

                YZ_I = Y_I
                g = 1j * Y_II
                inc_term = 1j * self.n_I * np.cos(self.theta) * delta_i0

                oneover_E_conv_all = np.zeros(len(E_conv_all))  # Dummy for TE case

            elif self.polarization == 1:  # TM
                Z_I = np.diag(k_I_z / (k0 * self.n_I ** 2))
                Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))

                YZ_I = Z_I
                g = 1j * Z_II
                inc_term = 1j * delta_i0 * np.cos(self.theta) / self.n_I

                oneover_E_conv_all = self.permittivity_mapping(wl, oneover=True)

            else:
                raise ValueError

            T = np.eye(2 * self.fourier_order + 1)

            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                if self.polarization == 0:
                    A = Kx ** 2 - E_conv
                    eigenvalues, W = np.linalg.eig(A)
                    q = eigenvalues ** 0.5

                    Q = np.diag(q)
                    V = W @ Q

                elif self.polarization == 1:
                    E_i = np.linalg.inv(E_conv)
                    B = Kx @ E_i @ Kx - np.eye(E_conv.shape[0])
                    oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                    eigenvalues, W = np.linalg.eig(oneover_E_conv_i @ B)
                    q = eigenvalues ** 0.5

                    Q = np.diag(q)
                    V = oneover_E_conv @ W @ Q

                else:
                    raise ValueError

                X = np.diag(np.exp(-k0 * q * d))

                W_i = np.linalg.inv(W)
                V_i = np.linalg.inv(V)

                a = 0.5 * (W_i @ f + V_i @ g)
                b = 0.5 * (W_i @ f - V_i @ g)

                a_i = np.linalg.inv(a)

                f = W @ (np.eye(2 * self.fourier_order + 1) + X @ b @ a_i @ X)
                g = V @ (np.eye(2 * self.fourier_order + 1) - X @ b @ a_i @ X)
                T = T @ a_i @ X

            Tl = np.linalg.inv(g + 1j * YZ_I @ f) @ (1j * YZ_I @ delta_i0 + inc_term)
            R = f @ Tl - delta_i0
            T = T @ Tl

            DEri = R * np.conj(R) * np.real(k_I_z / (k0 * self.n_I * np.cos(self.theta)))
            if self.polarization == 0:
                DEti = T * np.conj(T) * np.real(k_II_z / (k0 * self.n_I * np.cos(self.theta)))
            elif self.polarization == 1:
                DEti = T * np.conj(T) * np.real(k_II_z / self.n_II ** 2) / (k0 * np.cos(self.theta) / self.n_I)
            else:
                raise ValueError

            # self.spectrum_r[i] = DEri.sum()
            # self.spectrum_t[i] = DEti.sum()
            self.spectrum_r = self.spectrum_r.at[i].set(DEri.sum())
            self.spectrum_t = self.spectrum_t.at[i].set(DEti.sum())

        return self.spectrum_r, self.spectrum_t

    def lalanne_1d_conical(self):

        if self.theta == 0:
            self.theta = 0.001

        # pmtvy = draw_1d(self.patterns)
        # E_conv_all = to_conv_mat(pmtvy, self.fourier_order)

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff).reshape((-1, 1))
        delta_i0[self.fourier_order] = 1

        I = np.eye(self.ff)
        O = np.zeros((self.ff, self.ff))

        for wl in self.wls:
            k0 = 2 * np.pi / wl

            E_conv_all = self.permittivity_mapping(wl)
            oneover_E_conv_all = self.permittivity_mapping(wl, oneover=True)

            kx_vector = k0 * (self.n_I * np.sin(self.theta) * np.cos(self.phi) - fourier_indices * (wl / self.period)).astype('complex')
            ky = k0 * self.n_I * np.sin(self.theta) * np.sin(self.phi)

            k_I_z = (k0 ** 2 * self.n_I ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5
            k_II_z = (k0 ** 2 * self.n_II ** 2 - kx_vector ** 2 - ky ** 2) ** 0.5

            k_I_z = k_I_z.conjugate()
            k_II_z = k_II_z.conjugate()

            varphi = np.arctan(ky / kx_vector)

            Y_I = np.diag(k_I_z / k0)
            Y_II = np.diag(k_II_z / k0)

            Z_I = np.diag(k_I_z / (k0 * self.n_I ** 2))
            Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))

            Kx = np.diag(kx_vector / k0)

            big_F = np.block([[I, O], [O, 1j * Z_II]])
            big_G = np.block([[1j * Y_II, O], [O, I]])

            big_T = np.eye(2 * self.ff)

            # for E_conv, d in zip(E_conv_all[::-1], self.thickness[::-1]):
            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):

                E_i = np.linalg.inv(E_conv)
                oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                A = Kx ** 2 - E_conv
                B = Kx @ E_i @ Kx - I
                A_i = np.linalg.inv(A)
                B_i = np.linalg.inv(B)

                to_decompose_W_1 = ky ** 2 * I + A
                to_decompose_W_2 = ky ** 2 * I + B @ oneover_E_conv_i

                # TODO: using eigh
                eigenvalues_1, W_1 = np.linalg.eig(to_decompose_W_1)
                eigenvalues_2, W_2 = np.linalg.eig(to_decompose_W_2)

                q_1 = eigenvalues_1 ** 0.5
                q_2 = eigenvalues_2 ** 0.5

                Q_1 = np.diag(q_1)
                Q_2 = np.diag(q_2)

                V_11 = A_i @ W_1 @ Q_1
                V_12 = (ky / k0) * A_i @ Kx @ W_2
                V_21 = (ky / k0) * B_i @ Kx @ E_i @ W_1
                V_22 = B_i @ W_2 @ Q_2

                X_1 = np.diag(np.exp(-k0 * q_1 * d))
                X_2 = np.diag(np.exp(-k0 * q_2 * d))

                F_c = np.diag(np.cos(varphi))
                F_s = np.diag(np.sin(varphi))

                V_ss = F_c @ V_11
                V_sp = F_c @ V_12 - F_s @ W_2
                W_ss = F_c @ W_1 + F_s @ V_21
                W_sp = F_s @ V_22
                W_ps = F_s @ V_11
                W_pp = F_c @ W_2 + F_s @ V_12
                V_ps = F_c @ V_21 - F_s @ W_1
                V_pp = F_c @ V_22

                big_I = np.eye(2 * (len(I)))
                big_X = np.block([[X_1, O], [O, X_2]])
                big_W = np.block([[V_ss, V_sp], [W_ps, W_pp]])
                big_V = np.block([[W_ss, W_sp], [V_ps, V_pp]])

                big_W_i = np.linalg.inv(big_W)
                big_V_i = np.linalg.inv(big_V)

                big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
                big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

                big_A_i = np.linalg.inv(big_A)

                big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
                big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

                big_T = big_T @ big_A_i @ big_X

            big_F_11 = big_F[:self.ff, :self.ff]
            big_F_12 = big_F[:self.ff, self.ff:]
            big_F_21 = big_F[self.ff:, :self.ff]
            big_F_22 = big_F[self.ff:, self.ff:]

            big_G_11 = big_G[:self.ff, :self.ff]
            big_G_12 = big_G[:self.ff, self.ff:]
            big_G_21 = big_G[self.ff:, :self.ff]
            big_G_22 = big_G[self.ff:, self.ff:]

            # Final Equation in form of AX=B
            final_A = np.block(
                [
                    [I, O, -big_F_11, -big_F_12],
                    [O, -1j * Z_I, -big_F_21, -big_F_22],
                    [-1j * Y_I, O, -big_G_11, -big_G_12],
                    [O, I, -big_G_21, -big_G_22],
                ]
            )

            final_B = np.block([
                [-np.sin(self.psi) * delta_i0],
                [-np.cos(self.psi) * np.cos(self.theta) * delta_i0],
                [-1j * np.sin(self.psi) * self.n_I * np.cos(self.theta) * delta_i0],
                [1j * self.n_I * np.cos(self.psi) * delta_i0]
            ]
            )

            final_X = np.linalg.inv(final_A) @ final_B

            R_s = final_X[:self.ff, :].flatten()
            R_p = final_X[self.ff:2 * self.ff, :].flatten()

            big_T = big_T @ final_X[2 * self.ff:, :]
            T_s = big_T[:self.ff, :].flatten()
            T_p = big_T[self.ff:, :].flatten()

            DEri = R_s * np.conj(R_s) * np.real(k_I_z / (k0 * self.n_I * np.cos(self.theta))) \
                   + R_p * np.conj(R_p) * np.real((k_I_z / self.n_I ** 2) / (k0 * self.n_I * np.cos(self.theta)))

            DEti = T_s * np.conj(T_s) * np.real(k_II_z / (k0 * self.n_I * np.cos(self.theta))) \
                   + T_p * np.conj(T_p) * np.real((k_II_z / self.n_II ** 2) / (k0 * self.n_I * np.cos(self.theta)))

            self.spectrum_r.append(DEri.sum())
            self.spectrum_t.append(DEti.sum())

        return self.spectrum_r, self.spectrum_t

    def lalanne_2d(self):

        if self.theta == 0:
            self.theta = 0.001

        # pmtvy = draw_1d(self.patterns)
        # E_conv_all = to_conv_mat(pmtvy, self.fourier_order)

        fourier_indices = np.arange(-self.fourier_order, self.fourier_order + 1)

        delta_i0 = np.zeros(self.ff**2).reshape((-1, 1))
        delta_i0[self.ff ** 2 // 2, 0] = 1

        I = np.eye(self.ff ** 2)
        O = np.zeros((self.ff ** 2, self.ff ** 2))

        center = self.ff ** 2

        for wl in self.wls:

            E_conv_all = self.permittivity_mapping(wl)
            oneover_E_conv_all = self.permittivity_mapping(wl, oneover=True)

            k0 = 2 * np.pi / wl

            kx_vector = k0 * (self.n_I * np.sin(self.theta) * np.cos(self.phi) - fourier_indices * (wl / self.period[0])).astype('complex')
            ky_vector = k0 * (self.n_I * np.sin(self.theta) * np.sin(self.phi) - fourier_indices * (wl / self.period[1])).astype('complex')

            Kx = np.diag(np.tile(kx_vector, self.ff).flatten()) / k0
            Ky = np.diag(np.tile(ky_vector.reshape((-1, 1)), self.ff).flatten()) / k0

            k_I_z = (k0 ** 2 * self.n_I ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5
            k_II_z = (k0 ** 2 * self.n_II ** 2 - kx_vector ** 2 - ky_vector.reshape((-1, 1)) ** 2) ** 0.5

            k_I_z = k_I_z.flatten().conjugate()
            k_II_z = k_II_z.flatten().conjugate()

            varphi = np.arctan(ky_vector.reshape((-1, 1)) / kx_vector).flatten()

            Y_I = np.diag(k_I_z / k0)
            Y_II = np.diag(k_II_z / k0)

            Z_I = np.diag(k_I_z / (k0 * self.n_I ** 2))
            Z_II = np.diag(k_II_z / (k0 * self.n_II ** 2))

            big_F = np.block([[I, O], [O, 1j * Z_II]])
            big_G = np.block([[1j * Y_II, O], [O, I]])

            big_T = np.eye(self.ff ** 2 * 2)

            for E_conv, oneover_E_conv, d in zip(E_conv_all[::-1], oneover_E_conv_all[::-1], self.thickness[::-1]):
                E_i = np.linalg.inv(E_conv)

                B = Kx @ E_i @ Kx - I
                D = Ky @ E_i @ Ky - I
                oneover_E_conv_i = np.linalg.inv(oneover_E_conv)

                S2_from_S = np.block(
                    [
                        [Ky ** 2 + B @ oneover_E_conv_i, Kx @ (E_i @ Ky @ E_conv - Ky)],
                        [Ky @ (E_i @ Kx @ oneover_E_conv_i - Kx), Kx ** 2 + D @ E_conv]
                    ])

                # TODO: using eigh
                eigenvalues, W = np.linalg.eig(S2_from_S)

                q = eigenvalues ** 0.5

                q_1 = q[:center]
                q_2 = q[center:]

                Q = np.diag(q)
                Q_i = np.linalg.inv(Q)
                U1_from_S = np.block(
                    [
                        [-Kx @ Ky, Kx ** 2 - E_conv],
                        [oneover_E_conv_i - Ky ** 2, Ky @ Kx]  # TODO Check x y order
                    ]
                )
                V = U1_from_S @ W @ Q_i

                W_11 = W[:center, :center]
                W_12 = W[:center, center:]
                W_21 = W[center:, :center]
                W_22 = W[center:, center:]

                V_11 = V[:center, :center]
                V_12 = V[:center, center:]
                V_21 = V[center:, :center]
                V_22 = V[center:, center:]

                X_1 = np.diag(np.exp(-k0 * q_1 * d))
                X_2 = np.diag(np.exp(-k0 * q_2 * d))

                F_c = np.diag(np.cos(varphi))
                F_s = np.diag(np.sin(varphi))

                W_ss = F_c @ W_21 - F_s @ W_11
                W_sp = F_c @ W_22 - F_s @ W_12
                W_ps = F_c @ W_11 + F_s @ W_21
                W_pp = F_c @ W_12 + F_s @ W_22

                V_ss = F_c @ V_11 + F_s @ V_21
                V_sp = F_c @ V_12 + F_s @ V_22
                V_ps = F_c @ V_21 - F_s @ V_11
                V_pp = F_c @ V_22 - F_s @ V_12

                big_I = np.eye(2 * (len(I)))
                big_X = np.block([[X_1, O], [O, X_2]])
                big_W = np.block([[W_ss, W_sp], [W_ps, W_pp]])
                big_V = np.block([[V_ss, V_sp], [V_ps, V_pp]])

                big_W_i = np.linalg.inv(big_W)
                big_V_i = np.linalg.inv(big_V)

                big_A = 0.5 * (big_W_i @ big_F + big_V_i @ big_G)
                big_B = 0.5 * (big_W_i @ big_F - big_V_i @ big_G)

                big_A_i = np.linalg.inv(big_A)

                big_F = big_W @ (big_I + big_X @ big_B @ big_A_i @ big_X)
                big_G = big_V @ (big_I - big_X @ big_B @ big_A_i @ big_X)

                big_T = big_T @ big_A_i @ big_X


            big_F_11 = big_F[:center, :center]
            big_F_12 = big_F[:center, center:]
            big_F_21 = big_F[center:, :center]
            big_F_22 = big_F[center:, center:]

            big_G_11 = big_G[:center, :center]
            big_G_12 = big_G[:center, center:]
            big_G_21 = big_G[center:, :center]
            big_G_22 = big_G[center:, center:]

            # Final Equation in form of AX=B
            final_A = np.block(
                [
                    [I, O, -big_F_11, -big_F_12],
                    [O, -1j * Z_I, -big_F_21, -big_F_22],
                    [-1j * Y_I, O, -big_G_11, -big_G_12],
                    [O, I, -big_G_21, -big_G_22],
                ]
            )

            final_B = np.block([
                [-np.sin(self.psi) * delta_i0],
                [-np.cos(self.psi) * np.cos(self.theta) * delta_i0],
                [-1j * np.sin(self.psi) * self.n_I * np.cos(self.theta) * delta_i0],
                [1j * self.n_I * np.cos(self.psi) * delta_i0]
            ]
            )

            final_X = np.linalg.inv(final_A) @ final_B

            R_s = final_X[:self.ff ** 2, :].flatten()
            R_p = final_X[self.ff ** 2:2 * self.ff ** 2, :].flatten()

            big_T = big_T @ final_X[2 * self.ff ** 2:, :]
            T_s = big_T[:self.ff ** 2, :].flatten()
            T_p = big_T[self.ff ** 2:, :].flatten()

            DEri = R_s * np.conj(R_s) * np.real(k_I_z / (k0 * n_I * np.cos(theta))) \
                   + R_p * np.conj(R_p) * np.real((k_I_z / n_I ** 2) / (k0 * n_I * np.cos(theta)))

            DEti = T_s * np.conj(T_s) * np.real(k_II_z / (k0 * n_I * np.cos(theta))) \
                   + T_p * np.conj(T_p) * np.real((k_II_z / n_II ** 2) / (k0 * n_I * np.cos(theta)))
            self.spectrum_r.append(DEri.sum())
            self.spectrum_t.append(DEti.sum())

        return self.spectrum_r, self.spectrum_t


if __name__ == '__main__':
    n_I = 1
    n_II = 1

    theta = 0
    phi = 0
    psi = 0

    fourier_order = 3
    period = [0.7, 0.7]

    wls = np.linspace(0.5, 2.3, 400)

    polarization = 1  # TE 0, TM 1

    # permittivity in grating layer
    # patterns = [[3.48, 1, 0.3], [3.48, 1, 0.3]]  # n_ridge, n_groove, fill_factor
    patterns = [['SILICON', 1, 0.3], ['SILICON', 1, 0.3]]  # n_ridge, n_groove, fill_factor
    thickness = [0.46, 0.66]

    polarization_type = 0

    res = LalanneBase(polarization_type, n_I, n_II, theta, phi, psi, fourier_order, period, wls, polarization, patterns, thickness)

    # res.lalanne_1d()
    # res.lalanne_1d_conical()
    res.lalanne_2d()

    plt.plot(res.wls, res.spectrum_r)
    plt.plot(res.wls, res.spectrum_t)
    plt.show()
