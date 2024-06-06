from cstr_params import *

class CstrEnv():
    def __init__(self, obs_flag, he=1e-4, hs=0.01, x0=np.zeros(2)):
        self.x = x0.copy()
        self.he = he
        self.hs = hs
        self.obs_flag = obs_flag
    
    def get_reward(self, xk, uk):
        # reward_x = 1.0 / (np.fabs(xk[0]) + 1.0) + 1.0 / (np.fabs(xk[1]) + 1.0)
        # # if self.obs_flag == '1':
        # #     reward_x = 1.0 / (np.fabs(xk[0]) + 1.0)
        # # elif self.obs_falg == '2':
        # #     reward_x = 1.0 / (np.fabs(xk[1]) + 1.0)

        # reward_u = 0.01 / (np.fabs(uk[0]) + 1.0) + 0.01 / (np.fabs(uk[1]) + 1.0)

        # return reward_x + reward_u

        reward_x = 3 * np.sqrt(np.fabs(xk[0])) + np.sqrt(np.fabs(xk[1]))
        reward_u = (uk**2 / np.array([20.0, 30.0])).sum()

        return reward_x + reward_u

    
    def get_new_state(self, xk, uk, add_noise=False, noise=None):
        xk1 = xk.copy()
        if not add_noise:
            for _ in range(int(self.hs / self.he)):
                xk1[0] += self.he * (F/V * (uk[0] + CA0s - xk1[0] - CAs) - k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2)
                xk1[1] += self.he * (F/V * (T0 - xk1[1] - Ts) + (-delta_H)/sigma/Cp * k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2 + (uk[1] * u2_scale  + Qs)/sigma/Cp/V)
        else:
            for _ in range(int(self.hs / self.he)):
                xk1[0] += self.he * (F/V * (uk[0] + CA0s - xk1[0] - CAs) - k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2) + self.he * noise[0]
                xk1[1] += self.he * (F/V * (T0 - xk1[1] - Ts) + (-delta_H)/sigma/Cp * k0 * np.exp(-E/R/(xk1[1] + Ts)) * (xk1[0] + CAs)**2 + (uk[1] * u2_scale  + Qs)/sigma/Cp/V) + self.he * noise[1]

        return xk1
    
    def reset(self, x0=None):
        if x0 is None:
            self.x = np.random.uniform([-x1_bd, -x2_bd], [x1_bd, x2_bd], size=(2,))
        else:
            self.x = x0.copy()
        
        if self.obs_flag == '1':
            return np.array([self.x[0]])
        elif self.obs_flag == '2':
            return np.array([self.x[1]])
        return self.x
    
    def step(self, uk, add_noise=False, noise=None):
        reward = self.get_reward(self.x, uk)
        new_state = self.get_new_state(self.x, uk, add_noise, noise)
        done = 0
        if abs(new_state[0]) > x1_bd or abs(new_state[1]) > x2_bd:
            done = 1
        elif abs(new_state[0]) < 0.001 and abs(new_state[1]) < 0.001:
            done = 2
        
        self.x = new_state.copy()

        if self.obs_flag == '1':
            new_state = np.array([new_state[0]])
        elif self.obs_flag == '2':
            new_state = np.array([new_state[1]])
        
        return new_state, reward, done
