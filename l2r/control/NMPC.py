import torch


class BicycleModel:
    # ref {https://www.researchgate.net/publication/271545759_The_3-DoF_bicycle_model_with_the_simplified_piecewise_linear_tire_model}
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

        # Learnable parameters related to the properties of the car
        if init_params is None:
            # l
            self.params = torch.ones(3, requires_grad=True)
        else:
            self.params = torch.tensor(init_params, requires_grad=True)

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

        # Normal tyre forces
        F_zf_normal = (
            self.sprung_mass * 9.81 * self.cg_b / (self.length)
            + self.unsprung_mass_f * 9.81
        )
        F_zr_normal = (
            self.sprung_mass * 9.81 * self.cg_a / (self.length)
            + self.unsprung_mass_r * 9.81
        )

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
