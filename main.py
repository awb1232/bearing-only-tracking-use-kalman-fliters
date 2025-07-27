from runner import *

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

def main():

    dt = 1
    maxt = 2000
    noise_mean = 0
    noise_std = 1
    Sensor = Point2D(np.array([0, 0, 90, 2]), mode='bdcv', acceleration=1, omega=1, maneuver='o')
    Target = Point2D(np.array([5000, 5000, 0, -5]), mode='xyvv', acceleration=1, omega=1)
    model = Model(Sensor, Target, dt=dt, maxt=maxt, brg_noise_mean=noise_mean, brg_noise_std=noise_std, q_x=0.01, q_y=0.01)

    algorithms = Algorithms(model)

    number = 100
    runner = Runner(algorithms)

    # 生成固定的蒙特卡洛仿真数据
    runner.generate_monte_carlo_data(number)
    x0 = np.array([4000.0, 4000.0, 0.0, -0.0])
    P0 = np.diag([100.0 ** 2, 100.0 ** 2, 1.0 ** 2, 50.0 ** 2])

    methods = {
        # 'ekf':'palegreen',
        'plkf': 'silver',
        # 'plkf_old': 'violet',
        # 'plkf_rad': 'cyan',
        'bcplkf': 'grey',
        # 'ivplkf': 'violet',
        # 'samivplkf':'cyan',

        # 'ukf_old': 'cyan',
        # 'ukf': 'violet',
        'fubkf': 'cyan',
        # 'ckf': 'cyan',
        #
        # 'frekf': 'forestgreen',
        # 'frplkf':'darkgrey',
        # 'frukf': 'darkviolet',
        # 'frckf': 'c',
        #
        # 'lsfrekf':'green',
        # 'lsfrplkf': 'grey',
        # 'lsfrukf': 'purple',
        # 'lsfrckf': 'darkcyan',

        # 'mle': 'orange',
        # 'lstsq': 'blue',
        # 'lstsq1': 'red',

    }

    for method, color in methods.items():
        runner.select_method(method)
        runner.run_monte_carlo(x0, P0, reverse_step=600, partical_rev_step=20, color=color)

    runner.visualize(x_ylim=2000,
                     y_ylim=2000,
                     vx_ylim=5,
                     vy_ylim=5,
                     pos_ylim=2000,
                     crs_ylim=10,
                     spd_ylim=5,
                     crlb_analysis=True,
                     subplot=False,
                     noise_analysis=True)  # 添加方位误差分析

if __name__ == '__main__':
    main()