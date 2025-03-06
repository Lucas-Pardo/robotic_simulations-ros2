import numpy as np

class Environment:
    def __init__(self, height=3):
        self.height = height
        r = np.random.uniform(2, 3)
        self.goal_position = np.random.choice([-r, r])
        self.boundaries = [[[-4, 0.625], [4, 0.625]], [[-4, -0.625], [-0.75, -0.625]], [[0.75, -0.625], [4, -0.625]],
                           [[-0.75, -0.625 - height], [-0.75, -0.625]], [[0.75, -0.625 - height], [0.75, -0.625]],
                           [[-0.75, -0.625 - height], [0.75, -0.625 - height]]]
        self.start_zone = [[-0.75, 0.75], [-0.625, -0.625 - height + 1.5]]

    def generate_start_position(self, diameter):
        x = np.random.uniform(self.start_zone[0][0] + diameter/2, self.start_zone[0][1] - diameter/2)
        y = np.random.uniform(self.start_zone[1][0] - diameter/2, self.start_zone[1][1] + diameter/2)
        return x, y
    
    def apf(self, x, y, goal=None, rho=5, k=1, eta_0=1.2, gamma=2, k_rep=0.1, nu=1):
        # Common implementation of paraboloidal close, conical away potential field
        e = [self.goal_position - x, -y]
        d = np.sqrt(e[0]**2 + e[1]**2)
        F_x = 0
        F_y = 0
        if d > 1e-6:
            if d <= rho:
                F_x += k * e[0]
                F_y += k * e[1]
            else:
                F_x += k * rho * e[0] / d
                F_y += k * rho * e[1] / d
            
        # Common implementation of repulsive potential field
        for b_pos in self.boundaries:
            if b_pos[0][0] == b_pos[1][0]:
                # Vertical boundary
                x_o = b_pos[0][0]
                y_o = np.clip(y, b_pos[0][1], b_pos[1][1])
            else:
                # Horizontal boundary
                y_o = b_pos[0][1]
                x_o = np.clip(x, b_pos[0][0], b_pos[1][0])
            r = np.sqrt((x - x_o)**2 + (y - y_o)**2)
            if r <= eta_0:
                fx = k_rep / r**2 * (1/r - 1/eta_0)**(gamma-1) * (x - x_o) / r
                fy = k_rep / r**2 * (1/r - 1/eta_0)**(gamma-1) * (y - y_o) / r
                aux = 1 #np.random.choice([-1, 1])
                # F_x += nu * fx + (1 - nu) * aux * fy
                # F_y += nu * fy - (1 - nu) * aux * fx
                F_x += fx + nu * aux * fy
                F_y += fy - nu * aux * fx
        return F_x, F_y

