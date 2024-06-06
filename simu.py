from cstr_params import *
from ddpg_model import AgentFc
from cstr_env import CstrEnv
from lmpc import *

SIGMA1 = 1.0    # std for w1
SIGMA2 = 40.0    # std for w2

def save_data(save_dir, name, x1, x2, u1, u2):
    np.save(os.path.join(save_dir, name + '_x1.npy'), x1)
    np.save(os.path.join(save_dir, name + '_x2.npy'), x2)
    np.save(os.path.join(save_dir, name + '_u1.npy'), u1)
    np.save(os.path.join(save_dir, name + '_u2.npy'), u2)

def ddpg_simu_with_noise(step, agent, env, x0, noise=None):
    obs = env.reset(x0)

    x1 = []
    x2 = []
    u1 = []
    u2 = []

    if noise is None:
        noise = np.zeros((step, 2))
        for i in range(step):
            noise[i] = np.random.normal([0.0, 0.0], [SIGMA1, SIGMA2])
    
    for i in range(step):
        act = agent.choose_action(obs, add_noise=False)

        x1.append(obs[0])
        x2.append(obs[1])
        u1.append(act[0])
        u2.append(act[1])
        
        new_state, reward, done = env.step(act, add_noise=True, noise=noise[i])
        obs = new_state

    return x1, x2, u1, u2

def lmpc_simu_with_noise(step, x0, noise):
    if noise is None:
        noise = np.zeros((step, 2))
        for i in range(step):
            noise[i] = np.random.normal([0.0, 0.0], [SIGMA1, SIGMA2])
    
    _x0 = list(x0)
    u1=[]
    u2=[]
    x1=[]
    x2=[]

    for i in tqdm(range(step)):
        x1.append(_x0[0])
        x2.append(_x0[1])
        
        _u1, _u2 = lmpc(_x0)
        _x0 = get_xk1(_x0, [_u1[0], _u2[0]], noise=noise[i])

        u1.append(_u1[0])
        u2.append(_u2[0])
    
    x1 = np.array([float(item) for item in x1])
    x2 = np.array([float(item) for item in x2])
    u1 = np.array([float(item) for item in u1])
    u2 = np.array([float(item) for item in u2])
    u2 /= 1e5

    return x1, x2, u1, u2

if __name__ == '__main__':
    checkpoint_dir = 'checkpoint'
    graph_save_dir = 'graphs'
    os.makedirs(graph_save_dir, exist_ok=True)

    # initial states
    # x0 = np.random.uniform([-x1_bd, -x2_bd], [x1_bd, x2_bd], size=(2,))
    x0s = [[-1.0, -25.0], [-1.0, 25.0], [1.0, -25.0], [1.0, 25.0]]

    env = CstrEnv('12', x0=x0s[0])
    agent = AgentFc(a_lr=1e-4, c_lr=1e-3, input_size=2, action_size=2,
                    input_bds=[x1_bd, x2_bd], action_bds=[u1_bd, u2_bd], max_reward=500,
                    action_noise_mu=[0.0, 0.0], action_noise_sigma=[0.3, 0.005],
                    action_noise_decay=0.5, tau=0.005, env=env, discount=0.98,
                    max_size=10000, layer1_size=400, layer2_size=300,
                    batch_size=1, chkpt_dir=checkpoint_dir)

    agent.load_models()

    STEPS = 50

    noise = np.zeros((STEPS, 2))
    for i in range(STEPS):
        noise[i] = np.random.normal([0.0, 0.0], [SIGMA1, SIGMA2])
    
    np.save(os.path.join(graph_save_dir, 'w1.npy'), noise[:, 0])
    np.save(os.path.join(graph_save_dir, 'w2.npy'), noise[:, 1])

    for i in tqdm(range(len(x0s))):
        x0 = np.array(x0s[i])
        x1, x2, u1, u2 = ddpg_simu_with_noise(step=STEPS, agent=agent, env=env, x0=x0, noise=noise)
        save_data(graph_save_dir, str(i) + '_ddpg', x1, x2, u1, u2)

        x1, x2, u1, u2 = lmpc_simu_with_noise(step=STEPS, x0=x0, noise=noise)
        save_data(graph_save_dir, str(i) + '_lmpc', x1, x2, u1, u2)
