from cstr_params import *
from ddpg_model import AgentFc
from cstr_env import CstrEnv


def plot_noise(save_dir):
    w1 = np.load(os.path.join(save_dir, 'w1.npy'))
    w2 = np.load(os.path.join(save_dir, 'w2.npy'))
    w1_fig_file = os.path.join(save_dir, 'w1.png')
    w2_fig_file = os.path.join(save_dir, 'w2.png')

    plt.figure()
    plt.plot(list(range(len(w1))), w1)
    plt.title('w1')
    plt.ylabel('w1')
    plt.xlabel('step')
    plt.savefig(w1_fig_file)
    plt.close()

    plt.figure()
    plt.plot(list(range(len(w2))), w2)
    plt.title('w2')
    plt.ylabel('w2')
    plt.xlabel('step')
    plt.savefig(w2_fig_file)
    plt.close()


def plot_single_var(idx, data_name, title, y_label, x_label, save_dir):
    ddpg_data = np.load(os.path.join(save_dir, str(idx) + '_ddpg_' + data_name + '.npy'))
    lmpc_data = np.load(os.path.join(save_dir, str(idx) + '_lmpc_' + data_name + '.npy'))
    fig_file = os.path.join(save_dir, str(idx) + '_' + data_name + '.png')
    step = len(ddpg_data)

    plt.figure()
    plt.plot(list(range(step)), ddpg_data, linestyle='-', color='blue', label='ddpg')
    plt.plot(list(range(step)), lmpc_data, linestyle='--', color='red', label='lmpc')
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(fig_file)
    plt.close()


def plot_trajectory(idx, save_dir):
    ddpg_x1 = np.load(os.path.join(save_dir, str(idx) + '_ddpg_x1.npy'))
    ddpg_x2 = np.load(os.path.join(save_dir, str(idx) + '_ddpg_x2.npy'))
    lmpc_x1 = np.load(os.path.join(save_dir, str(idx) + '_lmpc_x1.npy'))
    lmpc_x2 = np.load(os.path.join(save_dir, str(idx) + '_lmpc_x2.npy'))
    fig_file = os.path.join(save_dir, str(idx) + '_trajectory.png')

    plt.figure()
    plt.plot(ddpg_x1, ddpg_x2, linestyle='-', color='blue', label='ddpg')
    plt.plot(lmpc_x1, lmpc_x2, linestyle='--', color='red', label='lmpc')

    # plot stable region
    # Ellipse parameters
    center = (0, 0)  # Center coordinates
    width = 2.8
    height = 74.96

    # Create an array of angles
    theta = np.linspace(0, 2*np.pi, 100)

    # Parametric equations for the ellipse
    x = center[0] + width/2 * np.cos(theta)
    y = center[1] + height/2 * np.sin(theta)

    plt.plot(x, y, color='black')

    plt.title('trajectory ' + str(idx))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.legend()
    plt.savefig(fig_file)
    plt.close()


def plot_stable_region(save_dir):
    fig_file = os.path.join(save_dir, 'stable_region.png')

    plt.figure()
    # plot stable region
    # Ellipse parameters
    center = (0, 0)  # Center coordinates
    width = 2.8
    height = 74.96

    # Create an array of angles
    theta = np.linspace(0, 2*np.pi, 100)

    # Parametric equations for the ellipse
    x = center[0] + width/2 * np.cos(theta)
    y = center[1] + height/2 * np.sin(theta)

    plt.plot(x, y, color='black')

    # plot the manually set range for x1 and x2
    # Rectangle parameters
    left_bottom = (-1, -26)  # Coordinates of the bottom-left corner
    width = 2
    height = 52

    # Define the coordinates of the rectangle corners
    xx = [left_bottom[0], left_bottom[0] + width, left_bottom[0] + width, left_bottom[0], left_bottom[0]]
    yy = [left_bottom[1], left_bottom[1], left_bottom[1] + height, left_bottom[1] + height, left_bottom[1]]
    plt.plot(xx, yy, color='b')

    plt.title('stable region')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.savefig(fig_file)
    plt.close()


def plot_rewards_steps(save_dir):
    r = np.load('rewards.npy')
    s = np.load('steps.npy')
    rewards_fig_file = os.path.join(save_dir, 'rewards.png')
    steps_fig_file = os.path.join(save_dir, 'steps.png')
    
    plt.figure()
    plt.plot(range(0, len(r), 20), r[::20], color='purple')
    plt.title('rewards per episode in training')
    plt.ylabel('rewards')
    plt.xlabel('episode')
    plt.savefig(rewards_fig_file)
    plt.close()

    plt.figure()
    plt.plot(range(0, len(s), 20), s[::20], color='green')
    plt.title('steps per episode in training')
    plt.ylabel('steps')
    plt.xlabel('episode')
    plt.savefig(steps_fig_file)
    plt.close()


def plot_q_value(checkpoint_dir, save_dir):
    fig_file = os.path.join(save_dir, 'q_value.png')

    x1 = np.linspace(-1, 1, 20)
    x2 = np.linspace(-5.5, 5.5, 100)
    x1, x2 = np.meshgrid(x1, x2)

    # load model
    env = CstrEnv('12')
    agent = AgentFc(a_lr=1e-4, c_lr=1e-3, input_size=2, action_size=2,
                    input_bds=[x1_bd, x2_bd], action_bds=[u1_bd, u2_bd], max_reward=500,
                    action_noise_mu=[0.0, 0.0], action_noise_sigma=[0.3, 0.005],
                    action_noise_decay=0.5, tau=0.005, env=env, discount=0.98,
                    max_size=10000, layer1_size=400, layer2_size=300,
                    batch_size=1, chkpt_dir=checkpoint_dir)

    agent.load_models()

    q_value = np.zeros(x1.shape)
    for i in tqdm(range(x1.shape[0])):
        for j in range(x1.shape[1]):
            state = torch.tensor([x1[i][j], x2[i][j]]).reshape(1, -1)
            u = agent.actor(state)
            q_value[i][j] = agent.critic(state, u)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x1, x2, q_value, cmap='viridis')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Q value')
    plt.title('Q value learned by critic')
    plt.savefig(fig_file)
    plt.close()


if __name__ == '__main__':
    checkpoint_dir = 'checkpoint'
    graph_save_dir = 'graphs'
    
    plot_noise(save_dir=graph_save_dir)

    for i in tqdm(range(4)):
        plot_single_var(i, data_name='x1', title='x1', y_label='x1', x_label='step', save_dir=graph_save_dir)
        plot_single_var(i, data_name='x2', title='x2', y_label='x2', x_label='step', save_dir=graph_save_dir)
        plot_single_var(i, data_name='u1', title='u1', y_label='u1', x_label='step', save_dir=graph_save_dir)
        plot_single_var(i, data_name='u2', title='u2', y_label='u2', x_label='step', save_dir=graph_save_dir)

    plot_rewards_steps(graph_save_dir)

    for i in tqdm(range(4)):
        plot_trajectory(i, save_dir=graph_save_dir)

    plot_stable_region(save_dir=graph_save_dir)

    plot_q_value(checkpoint_dir=checkpoint_dir, save_dir=graph_save_dir)
