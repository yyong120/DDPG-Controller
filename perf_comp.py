from cstr_params import *
from ddpg_model import AgentFc
from cstr_env import CstrEnv
from lmpc import *
from simu import *

if __name__ == '__main__':
    episode = 20
    step = 50

    checkpoint_dir = 'checkpoint'
    env = CstrEnv('12')
    agent = AgentFc(a_lr=1e-4, c_lr=1e-3, input_size=2, action_size=2,
                    input_bds=[x1_bd, x2_bd], action_bds=[u1_bd, u2_bd], max_reward=500,
                    action_noise_mu=[0.0, 0.0], action_noise_sigma=[0.3, 0.005],
                    action_noise_decay=0.5, tau=0.005, env=env, discount=0.98,
                    max_size=10000, layer1_size=400, layer2_size=300,
                    batch_size=1, chkpt_dir=checkpoint_dir)

    agent.load_models()

    ddpg_time = 0.0
    ddpg_x1_error = 0.0
    ddpg_x2_error = 0.0
    lmpc_time = 0.0
    lmpc_x1_error = 0.0
    lmpc_x2_error = 0.0

    for e in tqdm(range(episode)):
        # x0 starts from small values
        x0 = np.random.uniform([-0.01, -0.25], [0.01, 0.25], size=(2,))

        noise = np.zeros((step, 2))
        for i in range(step):
            noise[i] = np.random.normal([0.0, 0.0], [SIGMA1, SIGMA2])

        start = time()
        x1, x2, u1, u2 = ddpg_simu_with_noise(step=step, agent=agent, env=env, x0=x0, noise=noise)
        ddpg_time += time() - start
        ddpg_x1_error += np.sum(np.abs(np.array(x1)))
        ddpg_x2_error += np.sum(np.abs(np.array(x2)))

        start = time()
        x1, x2, u1, u2 = lmpc_simu_with_noise(step=step, x0=x0, noise=noise)
        lmpc_time += time() - start
        lmpc_x1_error += np.sum(np.abs(np.array(x1)))
        lmpc_x2_error += np.sum(np.abs(np.array(x2)))

    num = episode * step
    ddpg_time /= num
    lmpc_time /= num
    ddpg_x1_error /= num
    ddpg_x2_error /= num
    lmpc_x1_error /= num
    lmpc_x2_error /= num

    print(f'ddpg_time: {ddpg_time}')
    print(f'ddpg_x1_error: {ddpg_x1_error}')
    print(f'ddpg_x2_error: {ddpg_x2_error}')
    print(f'lmpc_time: {lmpc_time}')
    print(f'lmpc_x1_error: {lmpc_x1_error}')
    print(f'lmpc_x2_error: {lmpc_x2_error}')

            