import tensorflow as tf
import numpy as np
import timeit

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class OneDBioPINN:
    # Initialize the class

    def __init__(self, X_measurement_vessel1,
                 X_measurement_vessel2,
                 X_measurement_vessel3,
                 X_measurement_vessel4,
                 X_measurement_vessel5,
                 X_measurement_vessel6,
                 X_measurement_vessel7,
                 A_training_vessel1, U_training_vessel1,
                 A_training_vessel2, U_training_vessel2,
                 A_training_vessel3, U_training_vessel3,
                 A_training_vessel4, U_training_vessel4,
                 A_training_vessel5, U_training_vessel5,
                 A_training_vessel6, U_training_vessel6,
                 A_training_vessel7, U_training_vessel7,
                 X_residual_vessel1,
                 X_residual_vessel2,
                 X_residual_vessel3,
                 X_residual_vessel4,
                 X_residual_vessel5,
                 X_residual_vessel6,
                 X_residual_vessel7,
                 T_residual, T_measurement, layers, bif_points, T_initial, lengths):


        # in m2
        self.A_01 = 1.5105405796990447e-05 # mean between 13.591786456490883E-06 and 16.61902513749001E-06 (resp. distal and prox)
        self.A_02 = 1.205633304704887e-05 # mean between 10.520879637606859E-06 and 13.591786456490883E-06
        self.A_03 = 5.982849049496403E-06
        self.A_04 = 6.245800354601867E-06
        self.A_05 = 2.895291789548353E-06
        self.A_06 = 1.4526724430199207E-06
        self.A_07 = 6.245800354601867E-06


        self.rho = 1040. # in kg/m3
        self.mu = 4e-03 # in Pa*s
        self.nu = self.mu/self.rho # in m2/s
        # I have transformed in m -> see mathematical formulation
        self.k = np.array([0.0865/1e-03, 2.0/1e-03, -2.253*1e-03]) # for computing R0 # old: [33.7e-03, 0.3, -0.9]

        self.lengths = lengths

        # R proximal in m
        self.R_prox_1 = 2.30*1e-03
        self.R_prox_2 = 2.08*1e-03
        self.R_prox_3 = 1.38*1e-03
        self.R_prox_4 = 1.41*1e-03
        self.R_prox_5 = 0.96*1e-03
        self.R_prox_6 = 0.68*1e-03
        self.R_prox_7 = 1.41*1e-03

        # R distal (from 3 to 7 equal to proximal) in m
        self.R_dist_1 = 2.08*1e-03
        self.R_dist_2 = 1.83*1e-03
        self.R_dist_3 = 1.38*1e-03
        self.R_dist_4 = 1.41*1e-03
        self.R_dist_5 = 0.96*1e-03
        self.R_dist_6 = 0.68*1e-03
        self.R_dist_7 = 1.41*1e-03



        # non-dim. phase

        self.U = 1e+1

        self.L = np.sqrt((1. / 7.) * (self.A_01 + self.A_02 + self.A_03 + self.A_04 \
                                      + self.A_05 + self.A_06 + self.A_07))
        self.T = self.L / self.U
        self.p0 = self.rho * self.U ** 2

        self.A0 = self.L ** 2

        X_measurement_vessel1 = X_measurement_vessel1 / self.L
        X_measurement_vessel2 = X_measurement_vessel2 / self.L
        X_measurement_vessel3 = X_measurement_vessel3 / self.L
        X_measurement_vessel4 = X_measurement_vessel4 / self.L
        X_measurement_vessel5 = X_measurement_vessel5 / self.L
        X_measurement_vessel6 = X_measurement_vessel6 / self.L
        X_measurement_vessel7 = X_measurement_vessel7 / self.L

        X_residual_vessel1 = X_residual_vessel1 / self.L
        X_residual_vessel2 = X_residual_vessel2 / self.L
        X_residual_vessel3 = X_residual_vessel3 / self.L
        X_residual_vessel4 = X_residual_vessel4 / self.L
        X_residual_vessel5 = X_residual_vessel5 / self.L
        X_residual_vessel6 = X_residual_vessel6 / self.L
        X_residual_vessel7 = X_residual_vessel7 / self.L

        T_residual = T_residual / self.T
        T_measurement = T_measurement / self.T
        T_initial = T_initial / self.T

        # Normalize inputs
        self.Xmean1, self.Xstd1 = X_residual_vessel1.mean(0), X_residual_vessel1.std(0)
        self.Xmean2, self.Xstd2 = X_residual_vessel2.mean(0), X_residual_vessel2.std(0)
        self.Xmean3, self.Xstd3 = X_residual_vessel3.mean(0), X_residual_vessel3.std(0)
        self.Xmean4, self.Xstd4 = X_residual_vessel4.mean(0), X_residual_vessel4.std(0)
        self.Xmean5, self.Xstd5 = X_residual_vessel5.mean(0), X_residual_vessel5.std(0)
        self.Xmean6, self.Xstd6 = X_residual_vessel6.mean(0), X_residual_vessel6.std(0)
        self.Xmean7, self.Xstd7 = X_residual_vessel7.mean(0), X_residual_vessel7.std(0)

        self.Tmean, self.Tstd = T_residual.mean(0), T_residual.std(0)

        self.jac_x1 = 1.0 / self.Xstd1
        self.jac_x2 = 1.0 / self.Xstd2
        self.jac_x3 = 1.0 / self.Xstd3
        self.jac_x4 = 1.0 / self.Xstd4
        self.jac_x5 = 1.0 / self.Xstd5
        self.jac_x6 = 1.0 / self.Xstd6
        self.jac_x7 = 1.0 / self.Xstd7

        self.jac_t = 1.0 / self.Tstd

        self.X_f1 = (X_residual_vessel1 - self.Xmean1) / self.Xstd1
        self.X_u1 = (X_measurement_vessel1 - self.Xmean1) / self.Xstd1

        self.X_f2 = (X_residual_vessel2 - self.Xmean2) / self.Xstd2
        self.X_u2 = (X_measurement_vessel2 - self.Xmean2) / self.Xstd2

        self.X_f3 = (X_residual_vessel3 - self.Xmean3) / self.Xstd3
        self.X_u3 = (X_measurement_vessel3 - self.Xmean3) / self.Xstd3

        self.X_f4 = (X_residual_vessel4 - self.Xmean4) / self.Xstd4
        self.X_u4 = (X_measurement_vessel4 - self.Xmean4) / self.Xstd4

        self.X_f5 = (X_residual_vessel5 - self.Xmean5) / self.Xstd5
        self.X_u5 = (X_measurement_vessel5 - self.Xmean5) / self.Xstd5

        self.X_f6 = (X_residual_vessel6 - self.Xmean6) / self.Xstd6
        self.X_u6 = (X_measurement_vessel6 - self.Xmean6) / self.Xstd6

        self.X_f7 = (X_residual_vessel7 - self.Xmean7) / self.Xstd7
        self.X_u7 = (X_measurement_vessel7 - self.Xmean7) / self.Xstd7

        self.T_u = (T_measurement - self.Tmean) / self.Tstd
        self.T_f = (T_residual - self.Tmean) / self.Tstd
        self.T_i = (T_initial - self.Tmean) / self.Tstd

        self.layers = layers

        self.A_u1 = A_training_vessel1
        self.u_u1 = U_training_vessel1

        self.A_u2 = A_training_vessel2
        self.u_u2 = U_training_vessel2

        self.A_u3 = A_training_vessel3
        self.u_u3 = U_training_vessel3

        self.A_u4 = A_training_vessel4
        self.u_u4 = U_training_vessel4

        self.A_u5 = A_training_vessel5
        self.u_u5 = U_training_vessel5

        self.A_u6 = A_training_vessel6
        self.u_u6 = U_training_vessel6

        self.A_u7 = A_training_vessel7
        self.u_u7 = U_training_vessel7

        X22_fm = bif_points[0]/self.L
        X23_fm = bif_points[0]/self.L
        X24_fm = bif_points[0]/self.L
        X44_fm = bif_points[1]/self.L
        X45_fm = bif_points[1]/self.L
        X47_fm = bif_points[1]/self.L

        bif_p22 = (X22_fm - self.Xmean2) / self.Xstd2
        bif_p23 = (X23_fm - self.Xmean3) / self.Xstd3
        bif_p24 = (X24_fm - self.Xmean4) / self.Xstd4
        bif_p44 = (X44_fm - self.Xmean4) / self.Xstd4
        bif_p45 = (X45_fm - self.Xmean5) / self.Xstd5
        bif_p47 = (X47_fm - self.Xmean7) / self.Xstd7

        X22 = bif_p22[0]
        X23 = bif_p23[0]
        X24 = bif_p24[0]
        X44 = bif_p44[0]
        X45 = bif_p45[0]
        X47 = bif_p47[0]

        # Initialize network weights and biases
        self.weights1, self.biases1 = self.initialize_NN(layers)
        self.weights2, self.biases2 = self.initialize_NN(layers)
        self.weights3, self.biases3 = self.initialize_NN(layers)
        self.weights4, self.biases4 = self.initialize_NN(layers)
        self.weights5, self.biases5 = self.initialize_NN(layers)
        self.weights6, self.biases6 = self.initialize_NN(layers)
        self.weights7, self.biases7 = self.initialize_NN(layers)

        # Define placeholders and computational graph
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self.X22_fm = tf.constant([X22], shape=[1024, 1], dtype=tf.float32) # 1024 points in the interface
        self.X23_fm = tf.constant([X23], shape=[1024, 1], dtype=tf.float32)
        self.X24_fm = tf.constant([X24], shape=[1024, 1], dtype=tf.float32)
        self.X44_fm = tf.constant([X44], shape=[1024, 1], dtype=tf.float32)
        self.X45_fm = tf.constant([X45], shape=[1024, 1], dtype=tf.float32)
        self.X47_fm = tf.constant([X47], shape=[1024, 1], dtype=tf.float32)

        self.A_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.A_u1.shape[1]))
        self.u_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.u_u1.shape[1]))

        self.A_u_tf2 = tf.placeholder(tf.float32, shape=(None, self.A_u2.shape[1]))
        self.u_u_tf2 = tf.placeholder(tf.float32, shape=(None, self.u_u2.shape[1]))

        self.A_u_tf3 = tf.placeholder(tf.float32, shape=(None, self.A_u3.shape[1]))
        self.u_u_tf3 = tf.placeholder(tf.float32, shape=(None, self.u_u3.shape[1]))

        self.A_u_tf4 = tf.placeholder(tf.float32, shape=(None, self.A_u4.shape[1]))
        self.u_u_tf4 = tf.placeholder(tf.float32, shape=(None, self.u_u4.shape[1]))

        self.A_u_tf5 = tf.placeholder(tf.float32, shape=(None, self.A_u5.shape[1]))
        self.u_u_tf5 = tf.placeholder(tf.float32, shape=(None, self.u_u5.shape[1]))

        self.A_u_tf6 = tf.placeholder(tf.float32, shape=(None, self.A_u6.shape[1]))
        self.u_u_tf6 = tf.placeholder(tf.float32, shape=(None, self.u_u6.shape[1]))

        self.A_u_tf7 = tf.placeholder(tf.float32, shape=(None, self.A_u7.shape[1]))
        self.u_u_tf7 = tf.placeholder(tf.float32, shape=(None, self.u_u7.shape[1]))

        self.X_u_tf1 = tf.placeholder(tf.float32, shape=(None, self.X_u1.shape[1]))
        self.X_u_tf2 = tf.placeholder(tf.float32, shape=(None, self.X_u2.shape[1]))
        self.X_u_tf3 = tf.placeholder(tf.float32, shape=(None, self.X_u3.shape[1]))
        self.X_u_tf4 = tf.placeholder(tf.float32, shape=(None, self.X_u4.shape[1]))
        self.X_u_tf5 = tf.placeholder(tf.float32, shape=(None, self.X_u5.shape[1]))
        self.X_u_tf6 = tf.placeholder(tf.float32, shape=(None, self.X_u6.shape[1]))
        self.X_u_tf7 = tf.placeholder(tf.float32, shape=(None, self.X_u7.shape[1]))

        self.t_u_tf = tf.placeholder(tf.float32, shape=(None, self.T_u.shape[1]))
        self.t_i_tf = tf.placeholder(tf.float32, shape=(None, self.T_i.shape[1]))

        self.X_f_tf1 = tf.placeholder(tf.float32, shape=(None, self.X_f1.shape[1]))
        self.X_f_tf2 = tf.placeholder(tf.float32, shape=(None, self.X_f2.shape[1]))
        self.X_f_tf3 = tf.placeholder(tf.float32, shape=(None, self.X_f3.shape[1]))
        self.X_f_tf4 = tf.placeholder(tf.float32, shape=(None, self.X_f4.shape[1]))
        self.X_f_tf5 = tf.placeholder(tf.float32, shape=(None, self.X_f5.shape[1]))
        self.X_f_tf6 = tf.placeholder(tf.float32, shape=(None, self.X_f6.shape[1]))
        self.X_f_tf7 = tf.placeholder(tf.float32, shape=(None, self.X_f7.shape[1]))

        self.t_f_tf = tf.placeholder(tf.float32, shape=(None, self.T_f.shape[1]))

        self.A_u_pred1, self.u_u_pred1, _ = self.neural_net_vessel1(self.X_u_tf1, self.t_u_tf)
        self.A_u_pred2, self.u_u_pred2, _ = self.neural_net_vessel2(self.X_u_tf2, self.t_u_tf)
        self.A_u_pred3, self.u_u_pred3, _ = self.neural_net_vessel3(self.X_u_tf3, self.t_u_tf)
        self.A_u_pred4, self.u_u_pred4, _ = self.neural_net_vessel4(self.X_u_tf4, self.t_u_tf)
        self.A_u_pred5, self.u_u_pred5, _ = self.neural_net_vessel5(self.X_u_tf5, self.t_u_tf)
        self.A_u_pred6, self.u_u_pred6, _ = self.neural_net_vessel6(self.X_u_tf6, self.t_u_tf)
        self.A_u_pred7, self.u_u_pred7, _ = self.neural_net_vessel7(self.X_u_tf7, self.t_u_tf)

        self.A_f_pred1, self.u_f_pred1, self.p_f_pred1 = self.neural_net_vessel1(self.X_f_tf1, self.t_f_tf)
        self.A_f_pred2, self.u_f_pred2, self.p_f_pred2 = self.neural_net_vessel2(self.X_f_tf2, self.t_f_tf)
        self.A_f_pred3, self.u_f_pred3, self.p_f_pred3 = self.neural_net_vessel3(self.X_f_tf3, self.t_f_tf)
        self.A_f_pred4, self.u_f_pred4, self.p_f_pred4 = self.neural_net_vessel4(self.X_f_tf4, self.t_f_tf)
        self.A_f_pred5, self.u_f_pred5, self.p_f_pred5 = self.neural_net_vessel5(self.X_f_tf5, self.t_f_tf)
        self.A_f_pred6, self.u_f_pred6, self.p_f_pred6 = self.neural_net_vessel6(self.X_f_tf6, self.t_f_tf)
        self.A_f_pred7, self.u_f_pred7, self.p_f_pred7 = self.neural_net_vessel7(self.X_f_tf7, self.t_f_tf)

        self.r_A1, self.r_u1, self.r_p1 = self.pinn_vessel1(self.X_f_tf1, self.t_f_tf)
        self.r_A2, self.r_u2, self.r_p2 = self.pinn_vessel2(self.X_f_tf2, self.t_f_tf)
        self.r_A3, self.r_u3, self.r_p3 = self.pinn_vessel3(self.X_f_tf3, self.t_f_tf)
        self.r_A4, self.r_u4, self.r_p4 = self.pinn_vessel4(self.X_f_tf4, self.t_f_tf)
        self.r_A5, self.r_u5, self.r_p5 = self.pinn_vessel5(self.X_f_tf5, self.t_f_tf)
        self.r_A6, self.r_u6, self.r_p6 = self.pinn_vessel6(self.X_f_tf6, self.t_f_tf)
        self.r_A7, self.r_u7, self.r_p7 = self.pinn_vessel7(self.X_f_tf7, self.t_f_tf)

        self.loss_A1, self.loss_u1 = self.compute_measurement_loss_vessel1(self.A_u_pred1, self.u_u_pred1)
        self.loss_rA1, self.loss_ru1, self.loss_rp1 = self.compute_residual_loss_vessel1(self.r_A1, self.r_u1,
                                                                                         self.r_p1)

        self.loss_A2, self.loss_u2 = self.compute_measurement_loss_vessel2(self.A_u_pred2, self.u_u_pred2)
        self.loss_rA2, self.loss_ru2, self.loss_rp2 = self.compute_residual_loss_vessel2(self.r_A2, self.r_u2,
                                                                                         self.r_p2)

        self.loss_A3, self.loss_u3 = self.compute_measurement_loss_vessel3(self.A_u_pred3, self.u_u_pred3)
        self.loss_rA3, self.loss_ru3, self.loss_rp3 = self.compute_residual_loss_vessel3(self.r_A3, self.r_u3,
                                                                                         self.r_p3)

        self.loss_A4, self.loss_u4 = self.compute_measurement_loss_vessel4(self.A_u_pred4, self.u_u_pred4)
        self.loss_rA4, self.loss_ru4, self.loss_rp4 = self.compute_residual_loss_vessel4(self.r_A4, self.r_u4,
                                                                                         self.r_p4)

        self.loss_A5, self.loss_u5 = self.compute_measurement_loss_vessel5(self.A_u_pred5, self.u_u_pred5)
        self.loss_rA5, self.loss_ru5, self.loss_rp5 = self.compute_residual_loss_vessel5(self.r_A5, self.r_u5,
                                                                                         self.r_p5)

        self.loss_A6, self.loss_u6 = self.compute_measurement_loss_vessel6(self.A_u_pred6, self.u_u_pred6)
        self.loss_rA6, self.loss_ru6, self.loss_rp6 = self.compute_residual_loss_vessel6(self.r_A6, self.r_u6,
                                                                                         self.r_p6)

        self.loss_A7, self.loss_u7 = self.compute_measurement_loss_vessel7(self.A_u_pred7, self.u_u_pred7)
        self.loss_rA7, self.loss_ru7, self.loss_rp7 = self.compute_residual_loss_vessel7(self.r_A7, self.r_u7,
                                                                                         self.r_p7)

        self.loss_A = self.loss_A1 + self.loss_A2 + self.loss_A3 + self.loss_A4 + \
                      self.loss_A5 + self.loss_A6 + self.loss_A7

        self.loss_u = self.loss_u1 + self.loss_u2 + self.loss_u3 + self.loss_u4 + \
                      self.loss_u5 + self.loss_u6 + self.loss_u7

        self.loss_ru = self.loss_ru1 + self.loss_ru2 + self.loss_ru3 + self.loss_ru4 + \
                       self.loss_ru5 + self.loss_ru6 + self.loss_ru7

        self.loss_rA = self.loss_rA1 + self.loss_rA2 + self.loss_rA3 + self.loss_rA4 + \
                       self.loss_rA5 + self.loss_rA6 + self.loss_rA7

        self.loss_rp = self.loss_rp1 + self.loss_rp2 + self.loss_rp3 + self.loss_rp4 + \
                       self.loss_rp5 + self.loss_rp6 + self.loss_rp7

        self.loss_measurements = self.loss_A + self.loss_u

        self.loss_interface = self.compute_interface_loss()

        self.loss_residual = self.loss_rA + self.loss_ru + self.loss_rp

        self.loss = self.loss_residual + self.loss_interface + self.loss_measurements
        
        # Initialize losses backups
        self.backup_loss_value = []
        self.backup_loss_A = []
        self.backup_loss_u = []
        self.backup_loss_r = []
        self.backup_loss_c = []

        # Define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.train_op = self.optimizer.minimize(self.loss)

        config = tf.ConfigProto(log_device_placement=True)
        self.sess = tf.Session(config=config)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Setup saver function
        self.saver = tf.train.Saver(max_to_keep=5)
        self.it_total = 0
        self.intermediate_backup = 1 # for backup every 10k

    def calculate_R0(self, x, r_prox, r_dist, length):
        """
        Method to calculate the reference radius :math:`R_0(x)`.
        The default model is

        .. math::
            R_0(x) = R_{prox} \\exp(\\log(R_{distal} / R_{prox})(x/L))

        :param x: (array or float) spatial point to calculate the reference diameter
        :return: (array or float) :math:`R_0(x)`
        """
        return r_prox * tf.math.exp(np.log(r_dist / r_prox) * (x / length)) # you can't mix numpy and tf

    def f(self, R0):
        """
        This property calculates the f(r0, k) function

        .. math: `f(R_0, k) = \\frac{4}{3}(k_1 e^{R_0 k_2} + k_3)`

        :param R0: Reference diameter
        :type priority: float or ndarray[ndim=1, type=float]
        :param k: the empirical k params
        :type k: ndarray[ndim=1, type=float]
        :return: float or ndarray[ndim=1, type=float]
        """
        k1, k2, k3 = self.k
        res = (4 / 3.) * (k2 * tf.math.exp(k3 * R0) + k1) # you can't mix numpy and tf

        return res

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, H, weights, biases, layers):
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def neural_net_vessel1(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights1, self.biases1, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def neural_net_vessel2(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights2, self.biases2, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def neural_net_vessel3(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights3, self.biases3, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def neural_net_vessel4(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights4, self.biases4, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def neural_net_vessel5(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights5, self.biases5, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def neural_net_vessel6(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights6, self.biases6, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def neural_net_vessel7(self, x, t):
        Au = self.neural_net(tf.concat([x, t], 1), self.weights7, self.biases7, self.layers)
        A = Au[:, 0:1]
        u = Au[:, 1:2]
        p = Au[:, 2:3]
        return tf.exp(A), u, p

    def compute_interface_loss(self):

        A22, u22, p22 = self.neural_net_vessel2(self.X22_fm, self.t_f_tf)  # A, q, p

        A23, u23, p23 = self.neural_net_vessel3(self.X23_fm, self.t_f_tf)  # A, q, p

        A24, u24, p24 = self.neural_net_vessel4(self.X24_fm, self.t_f_tf)  # A, q, p

        A44, u44, p44 = self.neural_net_vessel4(self.X44_fm, self.t_f_tf)  # A, q, p

        A45, u45, p45 = self.neural_net_vessel5(self.X45_fm, self.t_f_tf)  # A, q, p

        A47, u47, p47 = self.neural_net_vessel7(self.X47_fm, self.t_f_tf)  # A, q, p

        Q22 = A22 * u22
        Q23 = A23 * u23
        Q24 = A24 * u24

        Q44 = A44 * u44
        Q45 = A45 * u45
        Q47 = A47 * u47

        loss_mass = tf.reduce_mean(tf.square((Q22 - Q23 - Q24))) + \
                    tf.reduce_mean(tf.square((Q44 - Q45 - Q47)))

        p_22 = p22 + (0.5 * u22 ** 2)
        p_23 = p23 + (0.5 * u23 ** 2)
        p_24 = p24 + (0.5 * u24 ** 2)
        p_44 = p44 + (0.5 * u44 ** 2)
        p_45 = p45 + (0.5 * u45 ** 2)
        p_47 = p47 + (0.5 * u47 ** 2)

        loss_momentum = tf.reduce_mean(tf.square(p_22 - p_23)) + tf.reduce_mean(tf.square(p_22 - p_24)) + \
                        tf.reduce_mean(tf.square(p_44 - p_45)) + tf.reduce_mean(tf.square(p_44 - p_47))

        return loss_mass + loss_momentum

    def pinn_vessel1(self, x, t): # x_hat, t_hat

        A, u, p = self.neural_net_vessel1(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd1 + self.Xmean1 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_1, self.R_dist_1, self.lengths[0])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_01) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x1

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x1

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x1
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x1

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x + p_x  + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta) # r_u* see photo

        return r_A, r_u, r_p

    def pinn_vessel2(self, x, t):

        A, u, p = self.neural_net_vessel2(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd2 + self.Xmean2 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_2, self.R_dist_2, self.lengths[1])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_02) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x2

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x2

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x2
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x2

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x + p_x + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta)

        return r_A, r_u, r_p

    def pinn_vessel3(self, x, t):

        A, u, p = self.neural_net_vessel3(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd3 + self.Xmean3 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_3, self.R_dist_3, self.lengths[2])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_03) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x3

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x3

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x3
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x3

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x + p_x + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta)

        return r_A, r_u, r_p

    def pinn_vessel4(self, x, t):

        A, u, p = self.neural_net_vessel4(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd4 + self.Xmean4 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_4, self.R_dist_4, self.lengths[3])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_04) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x4

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x4

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x4
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x4

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x + p_x + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta)

        return r_A, r_u, r_p

    def pinn_vessel5(self, x, t):
        A, u, p = self.neural_net_vessel5(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd5 + self.Xmean5 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_5, self.R_dist_5, self.lengths[4])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_05) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x5

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x5

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x5
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x5

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x + p_x + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta)

        return r_A, r_u, r_p

    def pinn_vessel6(self, x, t):

        A, u, p = self.neural_net_vessel6(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd6 + self.Xmean6 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_6, self.R_dist_6, self.lengths[5])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_06) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x6

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x6

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x6
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x6

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x  + p_x + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta)

        return r_A, r_u, r_p

    def pinn_vessel7(self, x, t):

        A, u, p = self.neural_net_vessel7(x, t)  # \hat{A}, \hat{u}, \hat{p}

        x_star = x*self.Xstd7 + self.Xmean7 # I get the initial x before standardization
        t_star = t*self.Tstd + self.Tmean # same for t

        R0 = self.calculate_R0(x_star*self.L, self.R_prox_7, self.R_dist_7, self.lengths[6])
        #A0 = np.pi*R0**2
        f_r0 = self.f(R0)
        delta = tf.math.pow((self.nu * t_star*self.T) / (2. * np.pi), 0.5)  # set delta -> I have to destand. because t is under sqrt

        r_p = f_r0*(tf.math.sqrt(A*self.A0 / self.A_07) - 1)

        p_x = tf.gradients(p, x)[0] * self.jac_x7

        A_t = tf.gradients(A, t)[0] * self.jac_t
        A_x = tf.gradients(A, x)[0] * self.jac_x7

        u_t = tf.gradients(u, t)[0] * self.jac_t
        u_x = tf.gradients(u, x)[0] * self.jac_x7
        u_square_x = tf.gradients(u**2,x)[0] * self.jac_x7

        r_A = A_t + u * A_x + A * u_x  # r_A*
        r_u = u_t + 0.5*u_square_x + p_x + 2*np.pi*u*self.nu*R0/(A*self.L*self.U*delta)

        return r_A, r_u, r_p

    def compute_residual_loss_vessel1(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred1 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_residual_loss_vessel2(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred2 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_residual_loss_vessel3(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred3 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_residual_loss_vessel4(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred4 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_residual_loss_vessel5(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred5 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_residual_loss_vessel6(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred6 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_residual_loss_vessel7(self, r_A, r_u, r_p):

        loss_rA = tf.reduce_mean(tf.square(r_A))
        loss_ru = tf.reduce_mean(tf.square(r_u))

        loss_rp = tf.reduce_mean(tf.square((self.p_f_pred7 - r_p * (1 / self.p0))))

        return loss_rA, loss_ru, loss_rp

    def compute_measurement_loss_vessel1(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u1 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u1 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def compute_measurement_loss_vessel2(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u2 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u2 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def compute_measurement_loss_vessel3(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u3 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u3 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def compute_measurement_loss_vessel4(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u4 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u4 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def compute_measurement_loss_vessel5(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u5 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u5 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def compute_measurement_loss_vessel6(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u6 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u6 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def compute_measurement_loss_vessel7(self, A_u, u_u):

        loss_A = tf.reduce_mean(tf.square((self.A_u7 - A_u * self.A0) / self.A0))
        loss_u = tf.reduce_mean(tf.square((self.u_u7 - u_u * self.U) / self.U))

        return loss_A, loss_u

    def fetch_minibatch(self, X1_f, X2_f, X3_f, X4_f, X5_f, X6_f, X7_f, t_f, N_f_batch):
        N_f = X1_f.shape[0]
        idx_f = np.random.choice(N_f, N_f_batch, replace=False)
        X1_f_batch = X1_f[idx_f, :]
        X2_f_batch = X2_f[idx_f, :]
        X3_f_batch = X3_f[idx_f, :]
        X4_f_batch = X4_f[idx_f, :]
        X5_f_batch = X5_f[idx_f, :]
        X6_f_batch = X6_f[idx_f, :]
        X7_f_batch = X7_f[idx_f, :]

        t_f_batch = t_f[idx_f, :]
        return X1_f_batch, X2_f_batch, X3_f_batch, X4_f_batch, X5_f_batch, X6_f_batch, X7_f_batch, t_f_batch

    # Trains the model by minimizing the MSE loss
    def train(self, nIter=20000, learning_rate=1e-4):

        start_time = timeit.default_timer()

        for it in range(nIter):

            X1_f_batch, X2_f_batch, X3_f_batch, \
            X4_f_batch, X5_f_batch, X6_f_batch, X7_f_batch, T_f_batch = self.fetch_minibatch(self.X_f1,
                                                                                             self.X_f2,
                                                                                             self.X_f3,
                                                                                             self.X_f4,
                                                                                             self.X_f5,
                                                                                             self.X_f6,
                                                                                             self.X_f7,
                                                                                             self.T_f,
                                                                                             N_f_batch=1024)
            # Define a dictionary for associating placeholders with data

            tf_dict = {self.X_u_tf1: self.X_u1,
                       self.X_u_tf2: self.X_u2,
                       self.X_u_tf3: self.X_u3,
                       self.X_u_tf4: self.X_u4,
                       self.X_u_tf5: self.X_u5,
                       self.X_u_tf6: self.X_u6,
                       self.X_u_tf7: self.X_u7,
                       self.X_f_tf1: X1_f_batch,
                       self.X_f_tf2: X2_f_batch,
                       self.X_f_tf3: X3_f_batch,
                       self.X_f_tf4: X4_f_batch,
                       self.X_f_tf5: X5_f_batch,
                       self.X_f_tf6: X6_f_batch,
                       self.X_f_tf7: X7_f_batch,
                       self.t_f_tf: T_f_batch,
                       self.t_u_tf: self.T_u,
                       self.t_i_tf: self.T_i,
                       self.A_u_tf1: self.A_u1, self.u_u_tf1: self.u_u1,
                       self.A_u_tf2: self.A_u2, self.u_u_tf2: self.u_u2,
                       self.A_u_tf3: self.A_u3, self.u_u_tf3: self.u_u3,
                       self.A_u_tf4: self.A_u4, self.u_u_tf4: self.u_u4,
                       self.A_u_tf5: self.A_u5, self.u_u_tf5: self.u_u5,
                       self.A_u_tf6: self.A_u6, self.u_u_tf6: self.u_u6,
                       self.A_u_tf7: self.A_u7, self.u_u_tf7: self.u_u7,
                       self.learning_rate: learning_rate}

            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            # Print
            if it % 1 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value, loss_A, loss_u, loss_r, loss_c = self.sess.run([self.loss,
                                                                            self.loss_A,
                                                                            self.loss_u,
                                                                            self.loss_residual,
                                                                            self.loss_interface], tf_dict)

                print('It: %d, Loss: %.3e, Loss_A: %.3e, Loss_u: %.3e, Loss_r: %.3e,\
                                           Loss_int: %.3e, Time: %.2f' %
                      (it, loss_value, loss_A, loss_u, loss_r, loss_c, elapsed))
                start_time = timeit.default_timer()
                
                # Save residuals each 100 iterations
                if self.it_total % 100 == 0:
                    self.store_losses(loss_value, loss_A, loss_u, loss_r, loss_c)

                start_time = timeit.default_timer()

            # Save backup if requested
            if self.intermediate_backup == 1:
                if self.it_total % 10000 == 0:
                    self.saver.save(self.sess, './Model/backup', global_step=self.it_total)

                    # param = [self.Xmean1, self.Xstd1, self.Xmean2, self.Xstd2, self.Xmean3, self.Xstd3, self.Xmean4, self.Xstd4, self.Xmean5, self.Xstd5, self.Xmean6, self.Xstd6, self.Xmean7, self.Xstd7, self.Tmean, self.Tstd]
                    # param_dict = dict({'Parameters': param})
                    # np.save('./Model/Parameters.npy', param_dict)

    # Evaluates predictions at test points
    def predict_vessel1(self, X1, t, file_name='NO'):

        t_save = t
        x_save = X1

        X1 = X1 / self.L
        t = t / self.T

        X1 = (X1 - self.Xmean1) / self.Xstd1
        t = (t - self.Tmean) / self.Tstd

        tf_dict1 = {self.X_f_tf1: X1, self.t_f_tf: t}

        A_star1 = self.sess.run(self.A_f_pred1, tf_dict1)
        u_star1 = self.sess.run(self.u_f_pred1, tf_dict1)
        p_star1 = self.sess.run(self.p_f_pred1, tf_dict1)

        A_star1 = A_star1 * self.A0
        u_star1 = u_star1 * self.U
        p_star1 = p_star1 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star1, u_star1, p_star1, file_name)

        return A_star1, u_star1, p_star1

    def predict_vessel2(self, X2, t, file_name='NO'):

        t_save = t
        x_save = X2

        X2 = X2 / self.L
        t = t / self.T

        X2 = (X2 - self.Xmean2) / self.Xstd2

        t = (t - self.Tmean) / self.Tstd
        tf_dict2 = {self.X_f_tf2: X2, self.t_f_tf: t}

        A_star2 = self.sess.run(self.A_f_pred2, tf_dict2)
        u_star2 = self.sess.run(self.u_f_pred2, tf_dict2)
        p_star2 = self.sess.run(self.p_f_pred2, tf_dict2)

        A_star2 = A_star2 * self.A0
        u_star2 = u_star2 * self.U
        p_star2 = p_star2 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star2, u_star2, p_star2, file_name)

        return A_star2, u_star2, p_star2

    def predict_vessel3(self, X3, t, file_name='NO'):

        t_save = t
        x_save = X3

        X3 = X3 / self.L
        t = t / self.T

        X3 = (X3 - self.Xmean3) / self.Xstd3
        t = (t - self.Tmean) / self.Tstd

        tf_dict3 = {self.X_f_tf3: X3, self.t_f_tf: t}

        A_star3 = self.sess.run(self.A_f_pred3, tf_dict3)
        u_star3 = self.sess.run(self.u_f_pred3, tf_dict3)
        p_star3 = self.sess.run(self.p_f_pred3, tf_dict3)

        A_star3 = A_star3 * self.A0
        u_star3 = u_star3 * self.U
        p_star3 = p_star3 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star3, u_star3, p_star3, file_name)

        return A_star3, u_star3, p_star3

    def predict_vessel4(self, X4, t, file_name='NO'):

        t_save = t
        x_save = X4

        X4 = X4 / self.L
        t = t / self.T

        X4 = (X4 - self.Xmean4) / self.Xstd4
        t = (t - self.Tmean) / self.Tstd

        tf_dict4 = {self.X_f_tf4: X4, self.t_f_tf: t}

        A_star4 = self.sess.run(self.A_f_pred4, tf_dict4)
        u_star4 = self.sess.run(self.u_f_pred4, tf_dict4)
        p_star4 = self.sess.run(self.p_f_pred4, tf_dict4)

        A_star4 = A_star4 * self.A0
        u_star4 = u_star4 * self.U
        p_star4 = p_star4 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star4, u_star4, p_star4, file_name)

        return A_star4, u_star4, p_star4

    def predict_vessel5(self, X5, t, file_name='NO'):

        t_save = t
        x_save = X5

        X5 = X5 / self.L
        t = t / self.T

        X5 = (X5 - self.Xmean5) / self.Xstd5
        t = (t - self.Tmean) / self.Tstd

        tf_dict5 = {self.X_f_tf5: X5, self.t_f_tf: t}

        A_star5 = self.sess.run(self.A_f_pred5, tf_dict5)
        u_star5 = self.sess.run(self.u_f_pred5, tf_dict5)
        p_star5 = self.sess.run(self.p_f_pred5, tf_dict5)

        A_star5 = A_star5 * self.A0
        u_star5 = u_star5 * self.U
        p_star5 = p_star5 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star5, u_star5, p_star5, file_name)

        return A_star5, u_star5, p_star5

    def predict_vessel6(self, X6, t, file_name='NO'):

        t_save = t
        x_save = X6

        X6 = X6 / self.L
        t = t / self.T

        X6 = (X6 - self.Xmean6) / self.Xstd6
        t = (t - self.Tmean) / self.Tstd

        tf_dict6 = {self.X_f_tf6: X6, self.t_f_tf: t}

        A_star6 = self.sess.run(self.A_f_pred6, tf_dict6)
        u_star6 = self.sess.run(self.u_f_pred6, tf_dict6)
        p_star6 = self.sess.run(self.p_f_pred6, tf_dict6)

        A_star6 = A_star6 * self.A0
        u_star6 = u_star6 * self.U
        p_star6 = p_star6 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star6, u_star6, p_star6, file_name)

        return A_star6, u_star6, p_star6

    def predict_vessel7(self, X7, t, file_name='NO'):

        t_save = t
        x_save = X7

        X7 = X7 / self.L
        t = t / self.T

        X7 = (X7 - self.Xmean7) / self.Xstd7
        t = (t - self.Tmean) / self.Tstd

        tf_dict7 = {self.X_f_tf7: X7, self.t_f_tf: t}

        A_star7 = self.sess.run(self.A_f_pred7, tf_dict7)
        u_star7 = self.sess.run(self.u_f_pred7, tf_dict7)
        p_star7 = self.sess.run(self.p_f_pred7, tf_dict7)

        A_star7 = A_star7 * self.A0
        u_star7 = u_star7 * self.U
        p_star7 = p_star7 * self.p0

        if file_name != 'NO':
            self.save_prediction(x_save, t_save, A_star7, u_star7, p_star7, file_name)

        return A_star7, u_star7, p_star7


    # Save the predictions in .npy files -------------------------------------------------------------------------------
    @staticmethod
    def save_prediction(x, t, A, u, p, file_name):

        pred = dict({'Position': x, 'Time': t, 'Area': A, 'Velocity': u, 'Pressure': p})
        path = './Results/' + file_name
        np.save(path, pred)
        
    def store_losses(self, loss_value, loss_A, loss_u, loss_r, loss_c):

        self.backup_loss_value.append(loss_value)
        self.backup_loss_A.append(loss_A)
        self.backup_loss_u.append(loss_u)
        self.backup_loss_r.append(loss_r)
        self.backup_loss_c.append(loss_c)

    def save_losses(self, path):

        losses_dict = dict({'Total_Loss': self.backup_loss_value, 'loss_area': self.backup_loss_A,
                           'loss_velo': self.backup_loss_u, 'loss_res': self.backup_loss_r,
                            'loss_cont': self.backup_loss_c})

        np.save('./Results/' + path + 'Losses.npy', losses_dict)
        
    def load_losses(self, path):

        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        
        losses = np.load('./Results/' + path + 'Losses.npy').item()
        Total_loss = losses['Total_Loss']
        loss_area = losses['loss_area']
        loss_velo = losses['loss_velo']
        loss_res = losses['loss_res']
        loss_cont = losses['loss_cont']

        # restore np.load for future normal usage
        np.load = np_load_old
        
        return Total_loss, loss_area, loss_velo, loss_res, loss_cont
        
    
    def save_NN(self, path, save_param=1):

        self.saver.save(self.sess, './Model/' + path + 'final model')

        self.save_losses(path)

        if save_param == 1:
            param = [self.Xmean1, self.Xstd1, self.Xmean2, self.Xstd2, self.Xmean3, self.Xstd3, self.Xmean4, self.Xstd4, self.Xmean5, self.Xstd5, self.Xmean6, self.Xstd6, self.Xmean7, self.Xstd7, self.Tmean, self.Tstd]
            param_dict = dict({'Parameters': param})
            np.save('./Model/' + path + 'Parameters.npy', param_dict)

    def load_NN(self, backup_file, path):

        # Restore weights and biases
        new_saver = tf.train.import_meta_graph(backup_file + '.meta')
        new_saver.restore(self.sess, backup_file)

        # save np.load
        np_load_old = np.load

        # modify the default parameters of np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


        
        param = np.load('./Model/' + path + 'Parameters.npy').item()
        self.Xmean1 = param['Parameters'][0]
        self.Xstd1 = param['Parameters'][1]
        self.Xmean2 = param['Parameters'][2]
        self.Xstd2 = param['Parameters'][3]
        self.Xmean3 = param['Parameters'][4]
        self.Xstd3 = param['Parameters'][5]
        self.Xmean4 = param['Parameters'][6]
        self.Xstd4 = param['Parameters'][7]
        self.Xmean5 = param['Parameters'][8]
        self.Xstd5 = param['Parameters'][9]
        self.Xmean6 = param['Parameters'][10]
        self.Xstd6 = param['Parameters'][11]
        self.Xmean7 = param['Parameters'][12]
        self.Xstd7 = param['Parameters'][13]

        # restore np.load for future normal usage
        np.load = np_load_old

        print('Restored session from ' + backup_file)

    # def save_NN(self, path, save_param=0):
    # 
    #     self.saver.save(self.sess, './Model/' + path + 'final_model')
    # 
    #     # Not necessary at the moment -> skip
    #     if save_param == 1:
    #         param = [self.Xmean1, self.Xstd1, self.Xmean2, self.Xstd2, self.Xmean3, self.Xstd3, self.Xmean4, self.Xstd4, self.Xmean5, self.Xstd5, self.Xmean6, self.Xstd6, self.Xmean7, self.Xstd7, self.Tmean, self.Tstd]
    #         param_dict = dict({'Parameters': param})
    #         np.save('./Model/' + path + 'Parameters.npy', param_dict)
    # 
    # def load_NN(self, file_name, param_file='DEFAULT'):
    # 
    #     #Set backup folder
    #     backup_folder = './Model/'
    #     backup_path = backup_folder + file_name
    # 
    #     # Restore weights and biases
    #     new_saver = tf.train.import_meta_graph(backup_path + '.meta')
    #     new_saver.restore(self.sess, backup_path)
    # 
    #     # --------------------------------------------------------------------
    #     # Restore parameters
    #     # save np.load
    #     np_load_old = np.load
    # 
    #     # modify the default parameters of np.load
    #     np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    # 
    #     if param_file == 'DEFAULT':
    #         param = np.load(backup_folder + 'Parameters.npy').item()
    #     else:
    #         param = np.load(backup_folder + param_file).item()
    # 
    #     self.Xmean1 = param['Parameters'][0]
    #     self.Xstd1 = param['Parameters'][1]
    #     self.Xmean2 = param['Parameters'][2]
    #     self.Xstd2 = param['Parameters'][3]
    #     self.Xmean3 = param['Parameters'][4]
    #     self.Xstd3 = param['Parameters'][5]
    #     self.Xmean4 = param['Parameters'][6]
    #     self.Xstd4 = param['Parameters'][7]
    #     self.Xmean5 = param['Parameters'][8]
    #     self.Xstd5 = param['Parameters'][9]
    #     self.Xmean6 = param['Parameters'][10]
    #     self.Xstd6 = param['Parameters'][11]
    #     self.Xmean7 = param['Parameters'][12]
    #     self.Xstd7 = param['Parameters'][13]
    # 
    #     self.Tmean = param['Parameters'][14]
    #     self.Tstd = param['Parameters'][15]
    # 
    #     # restore np.load for future normal usage
    #     np.load = np_load_old
    # 
    #     print('Restored session from ' + backup_path)