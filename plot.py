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

def plot_trajectory():
    pass

if __name__ == '__main__':
    graph_save_dir = 'graphs'
    
    plot_noise(save_dir=graph_save_dir)

    for i in tqdm(range(4)):
        plot_single_var(i, data_name='x1', title='x1', y_label='x1', x_label='step', save_dir=graph_save_dir)
        plot_single_var(i, data_name='x2', title='x2', y_label='x2', x_label='step', save_dir=graph_save_dir)
        plot_single_var(i, data_name='u1', title='u1', y_label='u1', x_label='step', save_dir=graph_save_dir)
        plot_single_var(i, data_name='u2', title='u2', y_label='u2', x_label='step', save_dir=graph_save_dir)
