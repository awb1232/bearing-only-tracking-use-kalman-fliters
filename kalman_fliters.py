import numpy as np

from angel_process import *

"""
说明：
1.假设你要进行n次卡尔曼滤波递推，那么observer_trajectory和measurements应该包含从初始状态和往后n个状态下的量测和坐标
2.observer_trajectory可以送入形状为{(n,2)}的二维位置坐标数组，也可以送入形状为{(n,4)}的四维状态坐标数组，但要保证前两个元素是x,y
3.噪声矩阵R取角度或弧度时对滤波器的性能有明显影响，对于EKF/UKF/CKF，三者在R取角度弧度时都能运行，但对于PLKF，R矩阵取角度时可以运行，取弧度时跟踪会发散
4.默认距离单位是米，速度单位是米/秒
5.这个代码没有考虑Q阵，因为仿真的时候被跟踪目标是按照严格的匀速直线运动设置并生成轨迹的
"""

class BearingOnlyEKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位拓展卡尔曼滤波EKF

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

    def _to_column_vector(self, x):
        """将(n,)形状的向量转换为(n,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(n,1)形状的列向量转换为(n,)形状的向量"""
        return x.flatten()

    def predict_bearing(self, x_col, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        返回以弧度表示的方位角
        x_col: 列向量形式的状态，形状为(4,1)
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x_col[0, 0] - observer_pos[0]
        dy = x_col[1, 0] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([[bearing]])  # 返回列向量形式

    def step(self):

        o_k = self.observer_trajectory[self.current_step]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)  # [4,1]

        # X(k|k-1)
        Xpre = self.F @ X   # [4,1]

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T   # [4,4]

        # H观测矩阵 - 使用列向量形式计算
        dx = Xpre[0, 0] - o_k[0]
        dy = Xpre[1, 0] - o_k[1]
        range_sq = dx**2 + dy**2

        H = np.array([[dy/range_sq, -dx/range_sq, 0, 0]])  # [1,4]

        # Z(k|k-1) 和 Z(k)
        Zpre = self.predict_bearing(Xpre)  # 列向量形式
        Z = np.array([[self.measurements[self.current_step][0]]])  # 列向量形式
        Z_residual_col = Z - Zpre

        # S(k)
        S = H @ self.P @ H.T + np.array([[self.R]])

        # K(k)
        K = Ppre @ H.T @ np.linalg.inv(S)   # [4,1]

        # X(k|k)
        Xest = Xpre + K @ Z_residual_col
        self.x = self._to_row_vector(Xest)  # 仅在最终输出时转换

        # P(k|k)
        self.P = Ppre - K @ H @ Ppre

        # 更新步数
        self.current_step += 1


class BearingOnlyPLKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位伪线性卡尔曼滤波器(PLKF)

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量

        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements）

    def _to_column_vector(self, x):
        """将(4,)形状的向量转换为(4,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(4,1)形状的列向量转换为(4,)形状的向量"""
        return x.flatten()

    def step_old(self):

        opos_k = self.observer_trajectory[self.current_step]
        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        # X(k-1|k-1)
        x_col = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre_col = self.F @ x_col

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1) - 使用列向量形式计算距离
        dx = Xpre_col[0, 0] - opos_k[0]
        dy = Xpre_col[1, 0] - opos_k[1]
        dst_pre_2 = dx**2 + dy**2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = np.array([[cos_z, -sin_z, 0, 0]])  # [1, 4]

        # S(k)
        S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])   # [1, 1]

        # K(k)
        K_col = Ppre @ H_pl.T @ np.linalg.inv(S)    # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = H_pl @ np.array([[opos_k[0]], [opos_k[1]], [0], [0]])  # 列向量形式
        Zpre_col = H_pl @ Xpre_col  # 列向量形式
        Z_residual_col = Zpl_k_col - Zpre_col

        # X(k|k)
        x_updated_col = Xpre_col + K_col @ Z_residual_col
        self.x = self._to_row_vector(x_updated_col)  # 仅在最终输出时转换

        # P(k|k)
        self.P = Ppre - K_col @ H_pl @ Ppre

        # 更新步数
        self.current_step += 1

    def step(self, Rdeg=True):

        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k) # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1) - 使用列向量形式计算距离
        # dx = Xpre[0, 0] - pos_sensor_k_col[0,0]
        # dy = Xpre[1, 0] - pos_sensor_k_col[1,0]
        # dst_pre_2 = dx ** 2 + dy ** 2

        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        if Rdeg:
            R_pl = dst_pre_2 * np.rad2deg(self.R)
        else:
            R_pl = dst_pre_2 * self.R

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        self.x = self._to_row_vector(Xest)

        # P(k|k)
        self.P = Ppre - K @ H_pl @ Ppre

        # 更新步数
        self.current_step += 1


class BearingOnlyBCPLKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位伪线性卡尔曼滤波器(PLKF)

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量

        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements）

    def _to_column_vector(self, x):
        """将(4,)形状的向量转换为(4,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(4,1)形状的列向量转换为(4,)形状的向量"""
        return x.flatten()

    def step(self):

        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k) # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1) - 使用列向量形式计算距离
        # dx = Xpre[0, 0] - pos_sensor_k_col[0,0]
        # dy = Xpre[1, 0] - pos_sensor_k_col[1,0]
        # dst_pre_2 = dx ** 2 + dy ** 2

        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        self.x = self._to_row_vector(Xest)

        # P(k|k)
        self.P = Ppre - K @ H_pl @ Ppre

        # 更新步数
        self.current_step += 1

    def bcplkf_step(self):
        # 于2025.7.16证明该bcplkf_step方法的偏差补偿相对普通的plkf_step方法有明显的性能提升
        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k)  # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1)
        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        # P(k|k)
        Pest = Ppre - K @ H_pl @ Ppre

        pos_replace = M @ Xest - pos_sensor_k_col   # [2,1]
        bias_compensation = Pest @ M.T @ pos_replace * (R_pl**-1) * self.R
        Xest_bc = Xest + bias_compensation

        self.x = self._to_row_vector(Xest_bc)  # 更新状态向量
        self.P = Pest

        self.current_step += 1  # 更新步数


class BearingOnlyIVPLKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位伪线性卡尔曼滤波器(PLKF)

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量

        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements）

    def _to_column_vector(self, x):
        """将(4,)形状的向量转换为(4,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(4,1)形状的列向量转换为(4,)形状的向量"""
        return x.flatten()

    def step(self):

        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k) # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1) - 使用列向量形式计算距离
        # dx = Xpre[0, 0] - pos_sensor_k_col[0,0]
        # dy = Xpre[1, 0] - pos_sensor_k_col[1,0]
        # dst_pre_2 = dx ** 2 + dy ** 2

        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        self.x = self._to_row_vector(Xest)

        # P(k|k)
        self.P = Ppre - K @ H_pl @ Ppre

        # 更新步数
        self.current_step += 1

    def bcplkf_step(self):
        # 于2025.7.16证明该bcplkf_step方法的偏差补偿相对普通的plkf_step方法有明显的性能提升
        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k)  # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1)
        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        # P(k|k)
        Pest = Ppre - K @ H_pl @ Ppre

        pos_replace = M @ Xest - pos_sensor_k_col   # [2,1]
        bias_compensation = Pest @ M.T @ pos_replace * (R_pl**-1) * self.R
        Xest_bc = Xest + bias_compensation

        self.x = self._to_row_vector(Xest_bc)  # 更新状态向量
        self.P = Pest

        self.current_step += 1  # 更新步数

    def ivplkf_step(self):

        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k)  # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1)
        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        # P(k|k)
        Pest = Ppre - K @ H_pl @ Ppre

        pos_replace = M @ Xest - pos_sensor_k_col  # [2,1]
        bias_compensation = Pest @ M.T @ pos_replace * (R_pl ** -1) * self.R
        Xest_bc = Xest + bias_compensation

        if self.current_step <= 600:
            self.x = self._to_row_vector(Xest_bc)
            self.P = Pest
        else:
            # p_bc(k|k)
            pos_est_bc = M @ Xest_bc    # [1,2]

            # theta_bc(k|k)
            bearing_est_bc = np.arctan2(
                pos_est_bc[0, 0] - pos_sensor_k_col[0, 0],
                pos_est_bc[1, 0] - pos_sensor_k_col[1, 0])
            cos_z_bc = np.cos(bearing_est_bc)
            sin_z_bc = np.sin(bearing_est_bc)


            # G(k)
            G = np.array([[cos_z_bc, -sin_z_bc]]) @ M   # [1,4]

            S_iv = H_pl @ Ppre @ G.T + R_pl
            K_iv = Ppre @ G.T @ np.linalg.inv(S_iv)

            Xest_iv = Xpre + K_iv @ Z_residual
            Pest_iv = Ppre - K_iv @ H_pl @ Ppre

            self.x = self._to_row_vector(Xest_iv)
            self.P = Pest_iv

        self.current_step += 1  # 更新步数

    def sam_ivplkf_step(self):
        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k)  # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1)
        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        # H(k) - 伪线性量测方程
        H_pl = u_k.T @ M  # [1, 4]

        # S(k)
        # S = H_pl @ Ppre @ H_pl.T + np.array([[R_pl]])  # [1, 1]
        S = H_pl @ Ppre @ H_pl.T + R_pl  # [1, 1]

        # K(k)
        K = Ppre @ H_pl.T @ np.linalg.inv(S)  # [4, 1]

        # Z(k|k-1) 和伪线性测量 Z(k)
        Zpl_k_col = u_k.T @ pos_sensor_k_col
        Zpre_col = H_pl @ Xpre  # 列向量形式
        Z_residual = Zpl_k_col - Zpre_col

        # X(k|k)
        Xest = Xpre + K @ Z_residual
        # P(k|k)
        Pest = Ppre - K @ H_pl @ Ppre

        pos_replace = M @ Xest - pos_sensor_k_col  # [2,1]
        bias_compensation = Pest @ M.T @ pos_replace * (R_pl ** -1) * self.R
        Xest_bc = Xest + bias_compensation

        if self.current_step <= 600:
            self.x = self._to_row_vector(Xest_bc)
            self.P = Pest
        else:
            # p_bc(k|k)
            pos_est_bc = M @ Xest_bc  # [1,2]

            # theta_bc(k|k)
            bearing_est_bc = np.arctan2(
                pos_est_bc[0, 0] - pos_sensor_k_col[0, 0],
                pos_est_bc[1, 0] - pos_sensor_k_col[1, 0])
            cos_z_bc = np.cos(bearing_est_bc)
            sin_z_bc = np.sin(bearing_est_bc)

            # G(k)
            G = np.array([[cos_z_bc, -sin_z_bc]]) @ M  # [1,4]

            S_iv = H_pl @ Ppre @ G.T + R_pl
            K_iv = Ppre @ G.T @ np.linalg.inv(S_iv)

            Xest_iv = Xpre + K_iv @ Z_residual
            Pest_iv = Ppre - K_iv @ H_pl @ Ppre


            if abs(rad1rad2sub(Z_k, bearing_est_bc)) <= 4*np.sqrt(self.R):
                X_final = Xest_iv
                P_final = Pest_iv
            else:
                X_final = Xest_bc
                P_final = Pest

            self.x = self._to_row_vector(X_final)
            self.P = P_final

        self.current_step += 1  # 更新步数


class BearingOnlyUKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位无迹卡尔曼滤波 UKF

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步

        # UKF参数
        self.alpha = 0.3  # 控制sigma点的散布程度
        self.beta = 2.0  # 先验分布的最优值 (2表示高斯分布)
        self.kappa = 0  # 次要参数，通常设为0 (x为单变量时设置为0 )

        # 计算缩放参数
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # 权重参数计算
        self.compute_weights()

    def _to_column_vector(self, x):
        """将(4,)形状的向量转换为(4,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(4,1)形状的列向量转换为(4,)形状的向量"""
        return x.flatten()

    def compute_weights(self):
        """计算sigma点的权重"""
        # 计算权重系数
        self.weights_m = np.zeros(2 * self.n + 1)  # 均值权重
        self.weights_c = np.zeros(2 * self.n + 1)  # 协方差权重

        # 中心点权重
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha ** 2 + self.beta)

        # 其余点权重
        for i in range(1, 2 * self.n + 1):
            self.weights_m[i] = 1 / (2 * (self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def generate_sigma_points_old(self):
        """生成sigma点"""
        # 计算矩阵平方根
        L = self.n + self.lambda_

        try:
            # 检查正定性
            if not np.all(np.linalg.eigvals(self.P) > 0):
                # 如果不是正定的，添加一个小的对角矩阵
                min_eig = np.min(np.linalg.eigvals(self.P))
                if min_eig < 0:
                    self.P -= 1.1 * min_eig * np.eye(self.n)

            sqrt_P = np.linalg.cholesky((L * self.P).astype(float))

            # 生成状态列向量
            x_col = self._to_column_vector(self.x)  # 形状为(n, 1)

            # 创建sigma点矩阵 - 每列是一个sigma点
            sigma_points_matrix = np.zeros((self.n, 2 * self.n + 1))  # 形状为(n, 2n+1)

            # 中心点
            sigma_points_matrix[:, 0] = x_col.flatten()

            # 其他sigma点
            for i in range(self.n):
                sigma_points_matrix[:, i + 1] = (x_col + sqrt_P[:, i:i+1]).flatten()
                sigma_points_matrix[:, i + 1 + self.n] = (x_col - sqrt_P[:, i:i+1]).flatten()

            # 转换为列向量形式返回，保持列向量格式
            sigma_points_cols = [self._to_column_vector(sigma_points_matrix[:, i]) for i in range(2 * self.n + 1)]

            return sigma_points_cols

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解作为备选方案
            print("警告：Cholesky分解失败，使用特征值分解代替")
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-6)  # 确保所有特征值为正
            sqrt_P = eigvecs @ np.diag(np.sqrt(L * eigvals))

            # 生成状态列向量
            x_col = self._to_column_vector(self.x)

            # 创建sigma点矩阵
            sigma_points_matrix = np.zeros((self.n, 2 * self.n + 1))

            # 中心点
            sigma_points_matrix[:, 0] = x_col.flatten()

            # 其他sigma点
            for i in range(self.n):
                sigma_points_matrix[:, i + 1] = (x_col + sqrt_P[:, i:i+1]).flatten()
                sigma_points_matrix[:, i + 1 + self.n] = (x_col - sqrt_P[:, i:i+1]).flatten()

            # 转换为列向量形式返回
            sigma_points_cols = [self._to_column_vector(sigma_points_matrix[:, i]) for i in range(2 * self.n + 1)]

            return sigma_points_cols

    def generate_sigma_points(self):
        """生成sigma点"""
        # 计算矩阵平方根
        L = self.n + self.lambda_

        try:
            # 检查正定性
            if not np.all(np.linalg.eigvals(self.P) > 0):
                # 如果不是正定的，添加一个小的对角矩阵
                min_eig = np.min(np.linalg.eigvals(self.P))
                if min_eig < 0:
                    self.P -= 1.1 * min_eig * np.eye(self.n)

            sqrt_P = np.linalg.cholesky((L * self.P).astype(float))

            # 生成状态列向量
            X = self._to_column_vector(self.x)  # 形状为(n, 1)

            # 直接生成列向量形式的sigma点列表
            sigma_points = []   # 应包含 2n + 1 个列向量

            # 中心点
            sigma_points.append(X.copy())

            # 其他sigma点
            for i in range(self.n):
                # 正向sigma点
                sigma_points.append(X + sqrt_P[:, i:i+1])
                # 负向sigma点
                sigma_points.append(X - sqrt_P[:, i:i+1])

            return sigma_points

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解作为备选方案
            print("警告：Cholesky分解失败，使用特征值分解代替")
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-6)  # 确保所有特征值为正
            sqrt_P = eigvecs @ np.diag(np.sqrt(L * eigvals))

            # 生成状态列向量
            X = self._to_column_vector(self.x)

            # 直接生成列向量形式的sigma点列表
            sigma_points = []

            # 中心点
            sigma_points.append(X.copy())

            # 其他sigma点 - 直接使用列向量计算
            for i in range(self.n):
                # 正向sigma点
                sigma_points.append(X + sqrt_P[:, i:i+1])
                # 负向sigma点
                sigma_points.append(X - sqrt_P[:, i:i+1])

            return sigma_points

    def state_transition(self, x_col, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x_col: 列向量形式的状态，形状为(4,1)
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x_col = F @ x_col

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            noise_col = self._to_column_vector(noise)
            new_x_col += noise_col

        return new_x_col

    def measurement_function(self, x_col, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        x_col: 列向量形式的状态，形状为(4,1)
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x_col[0, 0] - observer_pos[0]
        dy = x_col[1, 0] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def step_old(self):

        # 生成sigma采样点（已经是列向量形式）
        sigma_points_cols = self.generate_sigma_points_old()

        # 对各采样点进行状态转移计算
        sigma_points_pred_cols = [self.state_transition(sigma_col, add_noise=False) for sigma_col in sigma_points_cols]

        # X(k|k-1) - 使用列向量形式计算
        x_pred_col = np.zeros((self.n, 1))
        for i in range(len(sigma_points_pred_cols)):
            x_pred_col += self.weights_m[i] * sigma_points_pred_cols[i]

        # P(k|k-1) - 使用列向量形式计算协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred_cols)):
            diff_col = sigma_points_pred_cols[i] - x_pred_col
            P_pred += self.weights_c[i] * (diff_col @ diff_col.T)

        # Z(k|k-1) - 使用列向量形式计算测量
        z_pred = np.array([self.measurement_function(sigma_col) for sigma_col in sigma_points_pred_cols])
        z_mean = np.sum(self.weights_m.reshape(-1, 1) * z_pred, axis=0)

        # 计算测量预测协方差
        P_zz = 0
        for i in range(len(z_pred)):
            diff = rad1rad2sub1(z_pred[i], z_mean)
            P_zz += self.weights_c[i] * diff ** 2

        P_zz += self.R

        # 计算状态与测量的互相关矩阵 - 使用列向量形式
        P_xz_col = np.zeros((self.n, 1))
        for i in range(len(sigma_points_pred_cols)):
           diff_x_col = sigma_points_pred_cols[i] - x_pred_col
           diff_z = rad1rad2sub1(z_pred[i], z_mean)
           P_xz_col += self.weights_c[i] * diff_x_col * diff_z

        # 计算卡尔曼增益
        K_col = P_xz_col / P_zz

        Z = self.measurements[self.current_step]
        Z_residual = rad1rad2sub1(Z, z_mean)

        # X(k|k) - 使用列向量形式更新，然后转回行向量
        x_updated_col = x_pred_col + K_col * Z_residual[0]
        self.x = self._to_row_vector(x_updated_col)

        # P(k|k)
        self.P = P_pred - (K_col @ K_col.T) * P_zz

        self.current_step += 1

    def step(self):

        # 生成sigma采样点（已经是列向量形式）
        sigma_points_cols = self.generate_sigma_points()

        # 对各采样点进行状态转移计算
        sigma_points_pred_cols = [self.state_transition(sigma_col, add_noise=False) for sigma_col in sigma_points_cols]

        # X(k|k-1) - 使用列向量形式计算
        x_pred_col = np.zeros((self.n, 1))
        for i in range(len(sigma_points_pred_cols)):
            x_pred_col += self.weights_m[i] * sigma_points_pred_cols[i]

        # P(k|k-1) - 使用列向量形式计算协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred_cols)):
            diff_col = sigma_points_pred_cols[i] - x_pred_col
            P_pred += self.weights_c[i] * (diff_col @ diff_col.T)

        # Z(k|k-1) - 使用列向量形式计算测量
        z_pred = np.array([self.measurement_function(sigma_col) for sigma_col in sigma_points_pred_cols])
        z_mean = np.sum(self.weights_m.reshape(-1, 1) * z_pred, axis=0)

        # 计算测量预测协方差
        P_zz = 0
        for i in range(len(z_pred)):
            diff = rad1rad2sub1(z_pred[i], z_mean)
            P_zz += self.weights_c[i] * diff ** 2

        P_zz += self.R

        # 计算状态与测量的互相关矩阵 - 使用列向量形式
        P_xz_col = np.zeros((self.n, 1))
        for i in range(len(sigma_points_pred_cols)):
           diff_x_col = sigma_points_pred_cols[i] - x_pred_col
           diff_z = rad1rad2sub1(z_pred[i], z_mean)
           P_xz_col += self.weights_c[i] * diff_x_col * diff_z

        # 计算卡尔曼增益
        K_col = P_xz_col / P_zz

        Z = self.measurements[self.current_step]
        Z_residual = rad1rad2sub1(Z, z_mean)

        # X(k|k) - 使用列向量形式更新，然后转回行向量
        x_updated_col = x_pred_col + K_col * Z_residual[0]
        self.x = self._to_row_vector(x_updated_col)

        # P(k|k)
        self.P = P_pred - (K_col @ K_col.T) * P_zz

        self.current_step += 1


class BearingOnlyFUBKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位无迹卡尔曼滤波 UKF

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长
        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        if self.backward:
            # 逆向状态转移矩阵
            self.F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            self.F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步

        # UKF参数
        self.alpha = 0.3  # 控制sigma点的散布程度
        self.beta = 2.0  # 先验分布的最优值 (2表示高斯分布)
        self.kappa = 0  # 次要参数，通常设为0 (x为单变量时设置为0 )

        # 计算缩放参数
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n

        # 权重参数计算
        self.compute_weights()

    def _to_column_vector(self, x):
        """将(4,)形状的向量转换为(4,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(4,1)形状的列向量转换为(4,)形状的向量"""
        return x.flatten()

    def compute_weights(self):
        """计算sigma点的权重"""
        # 计算权重系数
        self.weights_m = np.zeros(2 * self.n + 1)  # 均值权重
        self.weights_c = np.zeros(2 * self.n + 1)  # 协方差权重

        # 中心点权重
        self.weights_m[0] = self.lambda_ / (self.n + self.lambda_)
        self.weights_c[0] = self.weights_m[0] + (1 - self.alpha ** 2 + self.beta)

        # 其余点权重
        for i in range(1, 2 * self.n + 1):
            self.weights_m[i] = 1 / (2 * (self.n + self.lambda_))
            self.weights_c[i] = self.weights_m[i]

    def generate_sigma_points(self):
        """生成sigma点"""
        # 计算矩阵平方根
        L = self.n + self.lambda_

        try:
            # 检查正定性
            if not np.all(np.linalg.eigvals(self.P) > 0):
                # 如果不是正定的，添加一个小的对角矩阵
                min_eig = np.min(np.linalg.eigvals(self.P))
                if min_eig < 0:
                    self.P -= 1.1 * min_eig * np.eye(self.n)

            sqrt_P = np.linalg.cholesky((L * self.P).astype(float))

            # 生成状态列向量
            X = self._to_column_vector(self.x)  # 形状为(n, 1)

            # 直接生成列向量形式的sigma点列表
            sigma_points = []   # 应包含 2n + 1 个列向量

            # 中心点
            sigma_points.append(X.copy())

            # 其他sigma点
            for i in range(self.n):
                # 正向sigma点
                sigma_points.append(X + sqrt_P[:, i:i+1])
                # 负向sigma点
                sigma_points.append(X - sqrt_P[:, i:i+1])

            return sigma_points

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解作为备选方案
            print("警告：Cholesky分解失败，使用特征值分解代替")
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-6)  # 确保所有特征值为正
            sqrt_P = eigvecs @ np.diag(np.sqrt(L * eigvals))

            # 生成状态列向量
            X = self._to_column_vector(self.x)

            # 直接生成列向量形式的sigma点列表
            sigma_points = []

            # 中心点
            sigma_points.append(X.copy())

            # 其他sigma点 - 直接使用列向量计算
            for i in range(self.n):
                # 正向sigma点
                sigma_points.append(X + sqrt_P[:, i:i+1])
                # 负向sigma点
                sigma_points.append(X - sqrt_P[:, i:i+1])

            return sigma_points

    def state_transition(self, x_col, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x_col: 列向量形式的状态，形状为(4,1)
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x_col = F @ x_col

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            noise_col = self._to_column_vector(noise)
            new_x_col += noise_col

        return new_x_col

    def measurement_function(self, x_col, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        x_col: 列向量形式的状态，形状为(4,1)
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x_col[0, 0] - observer_pos[0]
        dy = x_col[1, 0] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def step(self):
        M = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        Z_k = self.measurements[self.current_step][0]
        cos_z = np.cos(Z_k)
        sin_z = np.sin(Z_k)

        pos_sensor_k = self.observer_trajectory[self.current_step][:2]
        pos_sensor_k_col = self._to_column_vector(pos_sensor_k)  # [2,1]

        u_k = np.array([[cos_z],
                        [-sin_z]])  # [2,1]


        # 生成sigma采样点-包含2n+1个列向量的list
        sigma_points = self.generate_sigma_points()

        # 对各采样点进行状态转移计算
        sigma_points_pred = [self.state_transition(sigma_col, add_noise=False) for sigma_col in sigma_points]

        # X(k|k-1)
        Xpre_ukf = np.zeros((self.n, 1))
        for i in range(len(sigma_points_pred)):
            Xpre_ukf += self.weights_m[i] * sigma_points_pred[i]

        # P(k|k-1) - 使用列向量形式计算协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred)):
            diff_col = sigma_points_pred[i] - Xpre_ukf
            P_pred += self.weights_c[i] * (diff_col @ diff_col.T)

        # Z(k|k-1) - 使用列向量形式计算测量
        z_pred = np.array([self.measurement_function(sigma_col) for sigma_col in sigma_points_pred])
        z_mean = np.sum(self.weights_m.reshape(-1, 1) * z_pred, axis=0)

        # 计算测量预测协方差
        P_zz = 0
        for i in range(len(z_pred)):
            diff = rad1rad2sub1(z_pred[i], z_mean)
            P_zz += self.weights_c[i] * diff ** 2

        P_zz += self.R

        # 计算状态与测量的互相关矩阵 - 使用列向量形式
        P_xz_col = np.zeros((self.n, 1))
        for i in range(len(sigma_points_pred)):
           diff_x_col = sigma_points_pred[i] - Xpre_ukf
           diff_z = rad1rad2sub1(z_pred[i], z_mean)
           P_xz_col += self.weights_c[i] * diff_x_col * diff_z

        # 计算卡尔曼增益
        K_col = P_xz_col / P_zz

        Z = self.measurements[self.current_step]
        Z_residual = rad1rad2sub1(Z, z_mean)

        # X(k|k) - 使用列向量形式更新，然后转回行向量
        Xest_ukf = Xpre_ukf + K_col * Z_residual[0]

        Pest_ukf = P_pred - (K_col @ K_col.T) * P_zz

        """上面没有问题"""

        pos_est_ukf = M @ Xest_ukf  # [2,1]

        bearing_est_ukf = np.arctan2(
            pos_est_ukf[0, 0] - pos_sensor_k_col[0, 0],
            pos_est_ukf[1, 0] - pos_sensor_k_col[1, 0])

        cos_z_ukf = np.cos(bearing_est_ukf)
        sin_z_ukf = np.sin(bearing_est_ukf)

        dst_est_ukf = np.sqrt(
            (pos_est_ukf[0, 0] - pos_sensor_k_col[0, 0]) ** 2 +
            (pos_est_ukf[1, 0] - pos_sensor_k_col[1, 0]) ** 2)

        # G(k)
        G = np.array([[cos_z_ukf, -sin_z_ukf, 0, 0]])  # [1,4]
        dst_aux = np.array([[-sin_z_ukf, -cos_z_ukf, 0, 0]])@Xest_ukf - dst_est_ukf
        # dst_aux = np.array([[sin_z_ukf, cos_z_ukf, 0, 0]]) @ Xest_ukf - dst_est_ukf

        # z(k)伪线性测量
        Zpl_k_col = u_k.T @ pos_sensor_k_col

        # H(k)_
        H_ = G / dst_aux
        z_pl_ = Zpl_k_col / dst_aux

        # X(k-1|k-1)
        X = self._to_column_vector(self.x)

        # X(k|k-1)
        Xpre = self.F @ X

        # P(k|k-1)
        Ppre = self.F @ self.P @ self.F.T

        # d(k|k-1)
        dx = Xpre[0] - pos_sensor_k_col[0]
        dy = Xpre[1] - pos_sensor_k_col[1]
        dst_pre_2 = dx ** 2 + dy ** 2

        # n(k) - 伪线性噪声
        R_pl = dst_pre_2 * np.rad2deg(self.R)

        S_ = H_ @ Ppre @ H_.T + R_pl  # [1,1]

        K_fubkf = Ppre @ H_.T @ np.linalg.inv(S_)

        z_pred_fubkf = H_ @ Xpre
        z_residual_fubkf = z_pl_ - z_pred_fubkf
        Xest_fubkf = Xpre + K_fubkf @ (z_residual_fubkf)

        Pest_fubkf = Ppre - K_fubkf @ H_ @ Ppre

        # self.x = self._to_row_vector(Xest_ukf)
        #
        # self.P = Pest_ukf

        self.x = self._to_row_vector(Xest_fubkf)

        self.P = Pest_fubkf

        self.current_step += 1


class BearingOnlyCKF:

    def __init__(self, x0, P0, Q, R, dt, observer_trajectory, measurements, backward=False):
        """
        纯方位容积卡尔曼滤波 CKF

        :param x0: 初始状态向量 [x, y, vx, vy]
        :param P0: 初始化协方差矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差 (标量，弧度)
        :param dt: 时间步长（采样周期）
        :param observer_trajectory: 传感器轨迹，每行为一个时间步的位置 [x, y]，长度为n+1
        :param measurements: 量测方位序列（弧度）
        :param backward: 是否逆向滤波
        """

        self.n = len(x0)  # 状态维度
        self.x = x0.copy()  # 状态向量
        self.P = P0.copy()  # 协方差矩阵
        self.Q = Q.copy()  # 过程噪声协方差矩阵
        self.R = R  # 测量噪声协方差 (标量)
        self.dt = dt  # 时间步长

        self.backward = backward

        if self.backward:
            self.observer_trajectory = observer_trajectory[::-1]
            self.measurements = measurements[::-1]
        else:
            self.observer_trajectory = observer_trajectory
            self.measurements = measurements

        if len(self.observer_trajectory) != len(self.measurements):
            raise ValueError("传入的坐标序列和方位序列不等长")

        self.current_step = 1  # 当前步（用于索引observer_trajectory和measurements中对应元素）
        # 从 1 开始是因为初始量测值并不参与卡尔曼滤波迭代更新

        # CKF参数
        self.num_points = 2 * self.n  # CKF使用2n个立方点
        self.weight = 1.0 / self.num_points  # 所有点的权重相等

    def _to_column_vector(self, x):
        """将(4,)形状的向量转换为(4,1)形状的列向量"""
        return x.reshape(-1, 1)

    def _to_row_vector(self, x):
        """将(4,1)形状的列向量转换为(4,)形状的向量"""
        return x.flatten()

    def generate_cubature_points(self):
        """生成立方点"""
        # 计算矩阵平方根
        try:
            # 检查正定性
            if not np.all(np.linalg.eigvals(self.P) > 0):
                # 如果不是正定的，添加一个小的对角矩阵
                min_eig = np.min(np.linalg.eigvals(self.P))
                if min_eig < 0:
                    self.P -= 1.1 * min_eig * np.eye(self.n)

            sqrt_P = np.linalg.cholesky(self.P)

            # 生成单位方向向量矩阵 - 每列是一个方向向量
            # CKF使用2n个立方点，位于单位超球面上
            directions = np.zeros((self.n, 2 * self.n))  # 形状为(n, 2n)
            for i in range(self.n):
                directions[i, i] = 1.0  # 第i列为[0,...,1,0,...,0]^T
                directions[i, i + self.n] = -1.0  # 第(i+n)列为[0,...,-1,0,...,0]^T

            # 缩放方向向量
            scaled_directions = np.sqrt(self.n) * directions  # 形状为(n, 2n)

            # 生成状态列向量
            x_col = self._to_column_vector(self.x)  # 形状为(n, 1)

            # 生成立方点矩阵 - 每列是一个立方点
            cubature_points = x_col + sqrt_P @ scaled_directions  # 形状为(n, 2n)

            # 转换为列向量形式返回，保持列向量格式
            cubature_points_cols = [self._to_column_vector(cubature_points[:, i]) for i in range(2 * self.n)]

            return cubature_points_cols

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用特征值分解作为备选方案
            print("警告：Cholesky分解失败，使用特征值分解代替")
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-6)  # 确保所有特征值为正
            sqrt_P = eigvecs @ np.diag(np.sqrt(eigvals))

            # 生成单位方向向量矩阵
            directions = np.zeros((self.n, 2 * self.n))
            for i in range(self.n):
                directions[i, i] = 1.0
                directions[i, i + self.n] = -1.0

            # 缩放方向向量
            scaled_directions = np.sqrt(self.n) * directions

            # 生成状态列向量
            x_col = self._to_column_vector(self.x)

            # 生成立方点矩阵
            cubature_points = x_col + sqrt_P @ scaled_directions

            # 转换为列向量形式返回
            cubature_points_cols = [self._to_column_vector(cubature_points[:, i]) for i in range(2 * self.n)]

            return cubature_points_cols

    def state_transition(self, x_col, add_noise=False):
        """
        状态转移函数 - 匀速直线运动模型
        x_col: 列向量形式的状态，形状为(4,1)
        """
        if self.backward:
            # 逆向状态转移矩阵
            F = np.array([
                [1, 0, -self.dt, 0],
                [0, 1, 0, -self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 正向状态转移矩阵
            F = np.array([
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

        new_x_col = F @ x_col

        # 添加过程噪声
        if add_noise:
            noise = np.random.multivariate_normal(np.zeros(self.n), self.Q)
            noise_col = self._to_column_vector(noise)
            new_x_col += noise_col

        return new_x_col

    def measurement_function(self, x_col, step=None):
        """
        测量函数 - 计算从观测者到目标的方位角
        x_col: 列向量形式的状态，形状为(4,1)
        返回以弧度表示的方位角
        """
        if step is None:
            step = self.current_step

        observer_pos = self.observer_trajectory[step]
        dx = x_col[0, 0] - observer_pos[0]
        dy = x_col[1, 0] - observer_pos[1]
        bearing = np.arctan2(dx, dy)

        return np.array([bearing])

    def step(self):

        # 生成立方点（已经是列向量形式）
        cubature_points_cols = self.generate_cubature_points()

        # 传播立方点
        propagated_points_cols = [self.state_transition(x_col, add_noise=False) for x_col in cubature_points_cols]

        # X(k|k-1) - 使用列向量形式计算
        x_pred_col = np.zeros((self.n, 1))
        for i in range(len(propagated_points_cols)):
            x_pred_col += self.weight * propagated_points_cols[i]

        # P(k|k-1) - 使用列向量形式计算协方差
        P_pred = np.zeros((self.n, self.n))
        for i in range(len(propagated_points_cols)):
            P_pred += self.weight * (propagated_points_cols[i] @ propagated_points_cols[i].T)

        P_pred -= (x_pred_col @ x_pred_col.T)

        # 通过测量函数变换传播点
        z_points = np.array([self.measurement_function(x_col) for x_col in propagated_points_cols])

        # 计算预测测量均值
        z_pred = np.sum(z_points * self.weight, axis=0)

        # 计算预测测量协方差
        P_zz = 0
        for i in range(len(z_points)):
            diff = rad1rad2sub1(z_points[i], z_pred)
            P_zz += self.weight * diff[0] ** 2

        # 添加测量噪声协方差
        P_zz += self.R

        # 计算状态与测量的互相关矩阵 - 使用列向量形式
        P_xz_col = np.zeros((self.n, 1))
        for i in range(len(propagated_points_cols)):
            diff_x_col = propagated_points_cols[i] - x_pred_col
            diff_z = rad1rad2sub1(z_points[i], z_pred)
            P_xz_col += self.weight * diff_x_col * diff_z[0]

        # 计算卡尔曼增益
        K_col = P_xz_col / P_zz

        # 计算测量残差
        z = self.measurements[self.current_step]
        z_residual = rad1rad2sub1(z, z_pred)

        # X(k|k) - 使用列向量形式更新，然后转回行向量
        x_updated_col = x_pred_col + K_col * z_residual[0]
        self.x = self._to_row_vector(x_updated_col)

        # P(k|k)
        self.P = P_pred - (K_col @ K_col.T) * P_zz

        self.current_step += 1

