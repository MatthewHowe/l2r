from numpy.linalg import multi_dot
import torch

# Import the numpy package and name it np
import numpy as np
import math


class BicycleModel:
    # ref {https://www.researchgate.net/publication/271545759_The_3-DoF_bicycle_model_with_the_simplified_piecewise_linear_tire_model}
    # OR
    # {https://ieeexplore.ieee.org/document/6885617}
    def __init__(
        self,
        dt=1 / 20,
        n_state=4,  # state = [x, y, v, phi]
        n_ctrl=2,  # action = [a, delta]
        init_params=None,
    ):
        super().__init__()
        self.dt = dt
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.u_lower = torch.tensor([-1, -0.2]).float()
        self.u_upper = torch.tensor([1, 0.2]).float()

        # TODO: Get actual values for these
        self.sprung_mass = 1111  # kg, mass of car
        self.unsprung_mass_f = 60  # kg, mass of car
        self.unsprung_mass_r = 60  # kg, mass of car
        self.cg_a = 1.04  # Distance from front wheel to c.g.
        self.cg_b = 1.56  # Distance from rear wheel to c.g.
        self.cg_h = 0.25  # Height of center of mass
        self.length = self.cg_a + self.cg_b
        self.R_w = 0.28  # Radius of wheel

        self.C_nomx = 420000  # Nominal longitudinal stiffness
        self.C_nomy = 1000  # N/deg Nominal lateral stiffness
        self.s_nom1 = 0.01  # Nominal slip ratio 1
        self.s_nom2 = 0.25  # Nominal slip ratio 2
        self.mu = 1  # Friction coefficient

        # TODO: pretty sure this is fine, should check it
        # Normal tyre forces
        self.F_nomz_f = (
            self.sprung_mass * 9.81 * self.cg_b / (self.length)
            + self.unsprung_mass_f * 9.81
        )
        self.F_nomz_r = (
            self.sprung_mass * 9.81 * self.cg_a / (self.length)
            + self.unsprung_mass_r * 9.81
        )

        # Learnable parameters related to the properties of the car
        if init_params is None:
            # l
            self.params = torch.ones(3, requires_grad=True)
        else:
            self.params = torch.tensor(init_params, requires_grad=True)

    # def lateral_speed_calculation(self, u)

    def slip_ratio_calculations(self, u, v, r, delta_f, omega_f, omega_r):
        """
        Input:
            u: Longitudinal velocity
            v: Lateral velocity
            r: Yaw rate
            delta_f: Steering input
            omega_f: Front wheel speed (rad/s)
            omaga_r: Rear wheel speed (rad/s)
        Output:
            front_slip: (lateral, longitudinal)
            rear_slip: (lateral, longitudinal)
            
        """
        delta_r = 0  # No rear wheel steer

        u_wf = u * math.cos(delta_f) + (v + self.cg_a * r) * math.sin(delta_f)
        u_wr = u * math.cos(delta_r) + (v - self.cg_b * r) * math.sin(delta_r)

        # Lateral slip angle
        alpha_f = delta_f - (v + self.cg_a * r) / u_wf
        alpha_r = (self.cg_b * r - v) / u_wr

        # Longitudinal slip angle
        if u_wf < self.R_w * omega_f:
            s_wf = 1 - u_wf / (self.R_w * omega_f)
        else:
            s_wf = (self.R_w * omega_f) / u_wf - 1

        if u_wr < self.R_w * omega_r:
            s_wr = 1 - u_wr / (self.R_w * omega_r)
        else:
            s_wr = (self.R_w * omega_r) / u_wr - 1

        front_slip = (alpha_f, s_wf)
        rear_slip = (alpha_r, s_wf)

        return front_slip, rear_slip

    def magic_formula_approximation(self, F_z, F_nomz, s, alpha):
        """
        Input:
            F_z: Normal force on tyre
            s: longitudinal slip raio
            alpha = lateral slip ratio
            v: Lateral velocity
            r: Yaw rate
            delta_f: Steering input
            omega_f: Front wheel speed (rad/s)
            omaga_r: Rear wheel speed (rad/s)
        Output:
            front_slip: (lateral, longitudinal)
            rear_slip: (lateral, longitudinal)
            
        Page 107 (http://www.engineering108.com/Data/Engineering/Automobile/tyre-and-vehicle-dynamics.pdf)
        TODO - CHECK: I believe that sigma_x/y_max in the ref paper is sigma_sl the point which the tyre is 
        in full slip. This ends up just making a ratio in the calculations so could just be replaces with
        a learnable parameter or set to equal 1. 
        """
        mu = 0.8
        sigma_x_max = 1 / (2 * self.C_nomx * alpha ** 2) / (3 * mu * F_z)
        sigma_y_max = sigma_x_max

        sigma_x = s / (s + 1)
        sigma_x_star = sigma_x / sigma_x_max
        sigma_y = math.tan(alpha) / (s + 1)
        sigma_y_star = sigma_y / sigma_y_max
        sigma_total = math.sqrt(sigma_x ** 2 + sigma_y ** 2)
        sigma_total_star = math.sqrt(sigma_x_star ** 2 + sigma_y_star ** 2)

        s_mod = (sigma_total_star * sigma_x_max * np.sign(sigma_x)) / (
            1 + sigma_total_star * sigma_x_max * np.sign(sigma_x)
        )
        s_mod = s_mod * self.mu / mu
        # TODO: Use a atan2 or not??
        alpha_mod = math.atan2(sigma_total_star * sigma_y_max * np.sign(sigma_x))
        alpha_mod = alpha_mod * self.mu / mu

        F_x = (
            self.longitudinal_tyre_force(s_mod, F_z)
            * mu
            * (sigma_x / sigma_total)
            * math.sign(s_mod)
        )
        F_y = (
            self.lateral_tyre_force(alpha_mod, F_z)
            * mu
            * (sigma_y / sigma_total)
            * math.sign(alpha_mod)
        )

        return F_x, F_y

    def longitudinal_tyre_force(self, s, F_z, F_nomz):
        C_x = self.C_nomx * (1 + (F_z - F_nomz) / F_nomz)
        F_nomend = 3000 * (1 + (F_z - F_nomz) / (6 * F_nomz))
        F_end = (
            math.pow((F_nomend / F_nomz + (F_z - F_nomz) / (4 * F_nomz)), 0.8)
            * self.mu
            * F_z
        )
        s_1 = 7 * self.s_nom1
        s_2 = self.s_nom2 * math.pow(1 + (F_z - F_nomz) / (6 * F_nomz), -2)

        if abs(s) < s_1:
            F_x = C_x * s
        elif s_1 <= abs(s) < s_2:
            F_x = C_x * s_1 * np.sign(s_1)
        else:
            F_x = (
                C_x * s_1 - ((C_x * s_1 - F_end) / (s_2 - 1)) * (abs(s) - s_2)
            ) * np.sign(s_1)

        return F_x

    def lateral_tyre_force(self, alpha, F_z, F_nomz):
        F_peak = math.sqrt(0.9 - 1 - 0.182 * ((F_z / F_nomz) - 1))
        C_y = self.C_nomx * (1 + F_z / F_nomz) / 2.5
        alpha_0 = F_peak / C_y
        print(alpha_0, F_peak, C_y)

        if abs(alpha) < (0.85 * alpha_0):
            F_y = -C_y * alpha
        elif 0.85 * alpha_0 <= abs(alpha) < 1.75 * alpha_0:
            F_y = -(C_y / 6) * (abs(alpha) + 4.25 * alpha_0) * np.sign(alpha)
        else:
            # TODO: Something missing here need to figure out where this equation is wrong
            F_y = F_peak * np.sign(alpha)

        return F_y

    def forward(self, x_init, u):
        # k1 and k2 maps the action in [-1, 1] to actual acceleration and steering angle
        l, k1, k2 = torch.unbind(self.params)
        n = x_init.shape[0]

        # Align symbols to literature,
        # u = longitudinal velocity
        # v = lateral velocity
        # r = Yaw
        # TODO get lateral velocity from IMU?
        x, y, u, phi = torch.unbind(x_init, dim=-1)

        # Load transfer tyre forces, assume unsprung mass height at center of wheel
        F_zf_delta = (
            (u_dot - v * phi)
            * (self.mass * self.cg_h + self.unsprung_mass_f * self.R_w / 2)
        ) / self.length
        F_zr_delta = (
            (u_dot - v * phi)
            * (self.mass * self.cg_h + self.unsprung_mass_r * self.R_w / 2)
        ) / self.length

        F_zf = F_zf_normal - F_zf_delta
        F_zr = F_zr_normal + F_zr_delta

        # pdb.set_trace()
        a = u[:, 0] * k1
        delta = u[:, 1] * k2

        x_dot = torch.zeros(n, self.n_state)
        """
        ## w.r.t. center
        beta = torch.atan(0.5 * torch.tan(delta))
        x_dot[:, 0] = v*torch.cos(phi+beta)
        x_dot[:, 1] = v*torch.sin(phi+beta)
        x_dot[:, 2] = a
        x_dot[:, 3] = v*torch.sin(beta) / (l/2)

        """
        # w.r.t. the back axle
        x_dot[:, 0] = v * torch.cos(phi)
        x_dot[:, 1] = v * torch.sin(phi)
        x_dot[:, 2] = a
        x_dot[:, 3] = v * torch.tan(delta) / l

        return x_init + self.dt * x_dot

    def grad_input(self, x, u):
        """
        Input:
            x, u: (T-1, dim)
        Output:
            Ft: (T-1, m, m+n)
            ft: (T-1, m)
        """

        # k1 and k2 maps the action in [-1, 1] to actual acceleration and steering angle
        # l, k1, k2 = torch.unbind(self.params)
        l, k1, k2 = torch.unbind(self.params)

        T, _ = x.shape
        x, y, v, phi = torch.unbind(x, dim=-1)  # T

        # a = u[:, 0] * k1
        delta = u[:, 1] * k2
        """
        ## Reference: Center of Mass
        beta = torch.atan(0.5 * torch.tan(delta))

        A = torch.eye(self.n_state).repeat(T, 1, 1) # T x m x m
        A[:, 0, 2] = self.dt * torch.cos(phi+beta)
        A[:, 0, 3] = - self.dt * v * torch.sin(phi+beta)
        A[:, 1, 2] = self.dt * torch.sin(phi+beta)
        A[:, 1, 3] = self.dt * v * torch.cos(phi+beta)
        A[:, 3, 2] = self.dt * torch.sin(beta) / (l/2)

        B = torch.zeros(T, self.n_state, self.n_ctrl)
        partial_beta_delta = 0.5/(1+(0.5 * torch.tan(delta))**2)/(torch.cos(delta))**2
        B[:, 0, 1] = - self.dt * v * torch.sin(phi+beta) * partial_beta_delta
        B[:, 1, 1] = self.dt * v * torch.cos(phi+beta) * partial_beta_delta
        B[:, 2, 0] = self.dt
        B[:, 3, 1] = self.dt * v * torch.cos(beta)/(l/2) * partial_beta_delta
        """
        # Reference: Back Axle
        A = torch.eye(self.n_state).repeat(T, 1, 1)  # T x m x m
        A[:, 0, 2] = self.dt * torch.cos(phi)
        A[:, 0, 3] = -self.dt * v * torch.sin(phi)
        A[:, 1, 2] = self.dt * torch.sin(phi)
        A[:, 1, 3] = self.dt * v * torch.cos(phi)
        A[:, 3, 2] = self.dt * torch.tan(delta) / l

        B = torch.zeros(T, self.n_state, self.n_ctrl)
        B[:, 2, 0] = self.dt
        B[:, 3, 1] = self.dt * v / (l * torch.cos(delta) ** 2)

        # F = torch.cat([A, B], dim = -1) # T-1 x n_batch x m x (m+n)
        # pdb.set_trace()
        return A.squeeze(1), B.squeeze(1)

