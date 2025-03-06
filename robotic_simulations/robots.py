import numpy as np

class Robot:
    def __init__(self, x, y, u_x=0, u_y=0, diameter=1):
        self.x = x
        self.y = y
        self.u_x = u_x
        self.u_y = u_y
        self.diameter = diameter
        self.a = 2
        self.prev_pos = None
        self.prev_dh = None

    def update(self, dt):
        self.x += self.u_x * dt
        self.y += self.u_y * dt
    
    def apf_update(self, dt, env, goal=[0, 0], rho=5, k=1, eta_0=1.2, gamma=2, k_rep=1, nu=1):
        F_x, F_y = env.apf(self.x, self.y, goal, rho, k, eta_0, gamma, k_rep, nu)
        ### q_dot = F
        self.u_x = F_x
        self.u_y = F_y
        self.update(dt)
        
    def bb_grad_descent(self, env, tol=1e-6):
        dh = list(env.grad_h(self.x, self.y))
        if np.sqrt(dh[0]**2 + dh[1]**2) < tol:
            return self.x, self.y
        if self.prev_dh is not None:
            if np.sqrt((dh[0] - self.prev_dh[0])**2 + (dh[1] - self.prev_dh[1])**2) < tol:
                self.a = np.random.uniform(-25, -5)
            else:
                ### Geometric mean of a_long and a_short
                self.a = np.sqrt((self.x - self.prev_pos[0])**2 + (self.y - self.prev_pos[1])**2) / np.sqrt((dh[0] - self.prev_dh[0])**2 + (dh[1] - self.prev_dh[1])**2)
        x = self.x + self.a * dh[0]
        y = self.y + self.a * dh[1]
        self.prev_pos = [self.x, self.y]
        self.prev_dh = dh
        return x, y


class StickRobot(Robot):
    def __init__(self, x, y, theta=0, length=1.5, u_x=0, u_y=0, u_theta=0):
        super().__init__(x, y, u_x, u_y)
        self.length = length
        self.u_theta = u_theta
        self.theta = theta

    def update(self, dt):
        super().update(dt)
        self.theta += self.u_theta * dt
        
    def apf_update(self, dt, env, goal=[0, 0], rho=5, k=1, eta_0=1.2, gamma=2, k_rep=1, nu=1, mul_theta_k=2, mul_theta_eta=2, tail_orientation=0):
        x0 = self.x - self.length * np.cos(self.theta)
        y0 = self.y - self.length * np.sin(self.theta)
        F_x, F_y = env.apf(x0, y0, goal, rho, tail_orientation*k, mul_theta_eta*eta_0, gamma, mul_theta_k*k_rep, nu=nu/2)
        self.u_theta = (F_x * np.sin(self.theta) - F_y * np.cos(self.theta))
        self.u_theta = np.clip(self.u_theta, -np.pi/4, np.pi/4)
        super().apf_update(dt, env, goal, rho, k, eta_0, gamma, k_rep, nu)


