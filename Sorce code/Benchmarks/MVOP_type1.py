# Type1 mixed-variable optimization problems: discrete values are generated
# by Gaussian distribution in the [-80,80], expectation of the distribution
# is the shifted optimum.

import numpy as np

class MVOPT1(object):
    def __init__(self, prob_name, dim_r, dim_d, N_d):
        '''
        :param prob_name: problem name, including 'F1', 'F2', 'F3', 'F5'
        :param dim_r: dimension of continuous decision variables, including 5, 15, 25
        :param dim_d: dimension of discrete decision variables, including 5, 15, 25
        :param N_d: the number of values for discrete variables
        '''
        self.prob_name = prob_name
        self.r = dim_r
        self.dim = dim_r + dim_d

        self.N_lst = []
        self.N_d = N_d
        self.bounds = [-100, 100]

        # Shifted Elliposoid Function
        def f1(X):
            if isinstance(X, list): 
                X = np.array(X)[np.newaxis, :]

            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = tmp - self.x_shift
            f = np.sum(np.array([(i + 1) * z[:, i] ** 2 for i in range(self.dim)]), axis=0)
            return f

        # Shifted Rosenbrock’s Function(CEC2005 F6)
        def f2(X):
            if isinstance(X, list): 
                X = np.array(X)[np.newaxis, :]
            n = X.shape[0]
            # decode
            tmp = X.copy()
            for i in range(n):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = tmp - self.x_shift + 1
            f = np.zeros(n)
            for k in range(self.dim - 1):
                f += (100 * (z[:, k] ** 2 - z[:, k + 1]) ** 2 + (z[:, k] - 1) ** 2)
            return f

        # Shifted Alckey Function
        def f3(X):
            if isinstance(X, list): 
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = 0.32 * (tmp - self.x_shift)
            f = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt((1 / self.dim) * np.sum(z ** 2, axis=1))) - np.exp(
                (1 / self.dim) * np.sum(np.cos(2 * np.pi * z), axis=1))
            return f

        # Shifted Griewank Function
        def f4(X):
            if isinstance(X, list): 
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]
            if (X.ndim == 1):
                t = 1
            else:
                t = np.ones(X.shape[0])

            z = 6 * (tmp - self.x_shift)
            for i in range(self.dim):
                t *= np.cos(z[:, i] / np.sqrt(i + 1))
            f = 1 + (1 / 4000) * np.sum(z ** 2, axis=1) - t
            return f

        # Shifted Rastrigin Function
        def f5(X):
            if isinstance(X, list): 
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = 0.05 * (tmp - self.x_shift)
            f = np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, axis=1)
            return f

        # F1-F5: 10维
        if self.dim == 10:
            if self.prob_name == "F1":
                self.x_shift = np.array([[-16.1622, -45.7837, -77.875, 35.0299, -31.7985,
                                          -79.385, 71.875, -46.8745, 12.7822, -32.4944]])
                self.T = np.array([[-79.385, -68.7667, -78.2837, -60.9237, -76.1008],
                                   [71.875, 39.1584, 53.5349, 53.0347, 76.3195],
                                   [-46.8745, -68.9012, -38.8191, -29.6851, -58.8678],
                                   [12.7822, 2.6858, 17.8589, 28.2846, 23.3497],
                                   [-32.4944, -55.8217, -36.441, -44.1566, -26.1831]
                                   ])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f1

            elif self.prob_name == "F2":
                self.x_shift = np.array([[-67.1963, 50.2647, -55.8843, 78.5072, -74.0513,
                                          50.4688, 10.8691, -28.1226, -35.4082, -5.3238]])
                self.T = np.array([[50.4688, 52.8567, 49.287, 49.2159, 44.4857],
                                   [10.8691, -13.9148, 23.236, -1.042, 26.3276],
                                   [-28.1226, -43.3539, -10.6819, -32.6409, -40.1744],
                                   [-35.4082, -27.0611, -37.7903, -33.62, -39.2177],
                                   [-5.3238, -7.1207, 1.6733, -10.6555, -17.5576],
                                   ])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f2

            elif self.prob_name == "F3":
                self.x_shift = np.array([[-48.6963, -62.8892, -5.348, -8.6009, 32.1865,
                                          6.3813, -52.8817, 34.8827, -77.8583, -69.7131]])
                self.T = np.array([[6.3813, 15.5987, 6.737, 13.6742, 20.5048],
                                   [-52.8817, -56.5603, -58.5091, -48.485, -62.8261],
                                   [34.8827, 22.423, 24.5714, 36.3095, 52.3613],
                                   [-77.8583, -78.8671, -65.6888, -68.3704, -70.6715],
                                   [-69.7131, -63.9234, -58.1424, -56.8772, -60.4882]
                                   ])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f3

            elif self.prob_name == "F4":
                self.x_shift = np.array([[1.087, 32.8163, -47.3927, 61.2127, 65.9038,
                                          8.064, -20.0524, 60.8502, -59.4854, -31.2082]])
                self.T = np.array([[8.064, -6.4622, 8.0254, 2.4187, 4.6907],
                                   [-20.0524, -24.1022, -18.9675, -4.8144, -12.5602],
                                   [60.8502, 58.0555, 67.7075, 58.3066, 51.8559],
                                   [-59.4854, -59.5145, -51.4368, -57.3706, -51.7789],
                                   [-31.2082, -35.0085, -43.3999, -28.7323, -32.1823]
                                   ])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f4

            elif self.prob_name == "F5":
                self.x_shift = np.array([[37.7875, 25.6466, 75.8539, -10.1136, 41.1625,
                                          -19.7367, 19.7772, 66.4477, -72.4469, -37.3724]])
                self.T = np.array([[-19.7367, -12.5388, -10.8898, -21.4487, 4.229],
                                   [19.7772, 35.4497, 24.3298, 9.3878, 16.9281],
                                   [66.4477, 86.2022, 79.0859, 75.573, 59.0369],
                                   [-72.4469, -71.4305, -66.2036, -58.339, -78.8322],
                                   [-37.3724, -38.5753, -44.4339, -29.1662, -33.5218],
                                   ])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f5

        # F6-F10: 30维
        elif self.dim == 30:
            if self.prob_name == "F1":
                self.x_shift = np.array([[-16.1622, -45.7837, -77.875, 35.0299, -31.7985, -79.6745,
                                          -2.503, 20.6966, 10.9844, 46.4923, -39.5073, 6.5242,
                                          -27.6934, -23.4482, -1.346, -79.385, 71.875, -46.8745,
                                          12.7822, -32.4944, 49.6566, -48.0207, 43.2688, 76.0115,
                                          -21.5419, 45.669, 29.2559, -69.0237, 71.6123, 30.3036]])
                self.T = np.array([[-79.385, -68.7667, -78.2837, -60.9237, -76.1008],
                                   [71.875, 39.1584, 53.5349, 53.0347, 76.3195],
                                   [-46.8745, -68.9012, -38.8191, -29.6851, -58.8678],
                                   [12.7822, 2.6858, 17.8589, 28.2846, 23.3497],
                                   [-32.4944, -55.8217, -36.441, -44.1566, -26.1831],
                                   [49.6566, 28.571, 43.8424, 68.4411, 51.1256],
                                   [-48.0207, -51.2976, -32.2301, -39.5952, -49.2616],
                                   [43.2688, 36.6071, 42.8969, 45.6529, 41.4147],
                                   [76.0115, 78.5967, 78.7658, 57.6638, 87.027],
                                   [-21.5419, -30.0155, -14.4217, -21.9265, -30.9476],
                                   [45.669, 54.5476, 56.3059, 40.7178, 42.9398],
                                   [29.2559, 32.1074, 28.9266, 41.1001, 33.9031],
                                   [-69.0237, -79.2937, -75.882, -61.242, -74.3408],
                                   [71.6123, 61.6527, 71.306, 75.6912, 82.52],
                                   [30.3036, 47.8909, 31.9734, 33.7796, 29.0257]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f1

            elif self.prob_name == "F2":
                self.x_shift = np.array([[-67.1963, 50.2647, -55.8843, 78.5072, -74.0513, -36.5002,
                                          64.7148, 37.8788, 37.994, 19.6108, 27.7854, -27.5031,
                                          -64.4408, -61.0856, 40.4242, 50.4688, 10.8691, -28.1226,
                                          -35.4082, -5.3238, -56.9531, 55.5764, -37.4069, -49.3503,
                                          -13.8888, -22.6134, -9.0802, 33.8807, -43.8743, -32.7989]])
                self.T = np.array([[50.4688, 52.8567, 49.287, 49.2159, 44.4857],
                                   [10.8691, -13.9148, 23.236, -1.042, 26.3276],
                                   [-28.1226, -43.3539, -10.6819, -32.6409, -40.1744],
                                   [-35.4082, -27.0611, -37.7903, -33.62, -39.2177],
                                   [-5.3238, -7.1207, 1.6733, -10.6555, -17.5576],
                                   [-56.9531, -65.6662, -60.3521, -53.9814, -58.3596],
                                   [55.5764, 51.6299, 65.4225, 58.1889, 55.2519],
                                   [-37.4069, -43.3018, -10.8198, -36.677, -19.8214],
                                   [-49.3503, -44.7371, -42.3004, -46.4046, -61.0259],
                                   [-13.8888, -1.9341, -10.0653, -4.904, -24.9283],
                                   [-22.6134, -18.9599, -24.0118, -24.9135, -9.1539],
                                   [-9.0802, -7.1276, 2.4793, -0.3319, 0.4554],
                                   [33.8807, 60.6885, 46.7547, 24.648, 52.4062],
                                   [-43.8743, -33.1409, -32.8593, -30.1919, -43.9329],
                                   [-32.7989, -26.27, -42.3819, -30.6567, -42.2309]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f2

            elif self.prob_name == "F3":
                self.x_shift = np.array([[-48.6963, -62.8892, -5.348, -8.6009, 32.1865, 10.0334,
                                          -55.0711, -4.0932, -36.6631, -50.0353, -59.1566, 76.9305,
                                          -44.8734, 7.4912, -78.7536, 6.3813, -52.8817, 34.8827,
                                          -77.8583, -69.7131, 48.147, -60.6692, -25.8549, 72.8302,
                                          -37.8825, -27.5816, -4.2623, 77.475, 70.141, 42.0236]])
                self.T = np.array([[6.3813, 15.5987, 6.737, 13.6742, 20.5048],
                                   [-52.8817, -56.5603, -58.5091, -48.485, -62.8261],
                                   [34.8827, 22.423, 24.5714, 36.3095, 52.3613],
                                   [-77.8583, -78.8671, -65.6888, -68.3704, -70.6715],
                                   [-69.7131, -63.9234, -58.1424, -56.8772, -60.4882],
                                   [48.147, 67.9985, 32.1582, 38.1475, 49.4017],
                                   [-60.6692, -50.2903, -68.2733, -78.8651, -44.044],
                                   [-25.8549, -21.9518, -28.4456, -18.5534, -28.7794],
                                   [72.8302, 54.7231, 50.6423, 71.7053, 77.8392],
                                   [-37.8825, -45.1186, -41.6426, -15.9473, -34.9388],
                                   [-27.5816, -17.8879, -42.5414, -42.9613, -24.2074],
                                   [-4.2623, -0.4925, -6.97, 10.2182, 12.0672],
                                   [77.475, 67.831, 86.4626, 81.358, 95.5958],
                                   [70.141, 38.7039, 76.4972, 71.1255, 70.9277],
                                   [42.0236, 45.3878, 37.6246, 24.1552, 14.7688]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f3

            elif self.prob_name == "F4":
                self.x_shift = np.array([[1.087, 32.8163, -47.3927, 61.2127, 65.9038, -7.6319,
                                          28.5188, -23.7333, 28.3321, 45.3088, 24.5832, -55.2715,
                                          50.4791, -55.0356, -41.5454, 8.064, -20.0524, 60.8502,
                                          -59.4854, -31.2082, 15.371, -75.2107, -11.8859, 59.9726,
                                          27.1339, -30.8882, -1.6175, 55.2361, -76.3447, 9.2902]])
                self.T = np.array([[8.064, -6.4622, 8.0254, 2.4187, 4.6907],
                                   [-20.0524, -24.1022, -18.9675, -4.8144, -12.5602],
                                   [60.8502, 58.0555, 67.7075, 58.3066, 51.8559],
                                   [-59.4854, -59.5145, -51.4368, -57.3706, -51.7789],
                                   [-31.2082, -35.0085, -43.3999, -28.7323, -32.1823],
                                   [15.371, 21.4296, 40.3313, 24.1314, 2.9003],
                                   [-75.2107, -71.4888, -101.9311, -73.957, -66.1536],
                                   [-11.8859, -30.2464, -3.0793, -5.3866, -15.0466],
                                   [59.9726, 63.6583, 53.0353, 47.6211, 71.4752],
                                   [27.1339, 18.8899, 28.6499, 30.1178, 17.4997],
                                   [-30.8882, -25.7596, -37.1916, -41.2239, -36.8],
                                   [-1.6175, 8.6915, -26.7316, 3.1486, -6.4991],
                                   [55.2361, 41.8728, 41.2254, 36.4783, 69.3759],
                                   [-76.3447, -96.6821, -61.7104, -76.8758, -62.8834],
                                   [9.2902, 6.6974, 6.0637, 4.9247, 15.1033]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f4

            elif self.prob_name == "F5":
                self.x_shift = np.array([[37.7875, 25.6466, 75.8539, -10.1136, 41.1625, 50.5725,
                                          -19.3433, 58.1528, -0.7615, 32.019, -17.1938, 77.1054,
                                          44.7137, 78.5174, -28.8899, -19.7367, 19.7772, 66.4477,
                                          -72.4469, -37.3724, -40.9621, -77.7749, 46.4972, -14.3033,
                                          55.0064, -60.1448, 64.5772, 24.4927, 33.3941, -13.1568]])
                self.T = np.array([[-19.7367, -12.5388, -10.8898, -21.4487, 4.229],
                                   [19.7772, 35.4497, 24.3298, 9.3878, 16.9281],
                                   [66.4477, 86.2022, 79.0859, 75.573, 59.0369],
                                   [-72.4469, -71.4305, -66.2036, -58.339, -78.8322],
                                   [-37.3724, -38.5753, -44.4339, -29.1662, -33.5218],
                                   [-40.9621, -61.1055, -43.9516, -58.0527, -48.1514],
                                   [-77.7749, -60.2195, -90.8017, -79.9616, -87.8487],
                                   [46.4972, 67.596, 52.7707, 48.8682, 59.9521],
                                   [-14.3033, -17.5134, -6.2981, -11.5373, -2.5018],
                                   [55.0064, 41.2569, 51.1173, 61.5263, 56.5199],
                                   [-60.1448, -35.9869, -51.8534, -77.9464, -51.1539],
                                   [64.5772, 67.9773, 53.1607, 70.573, 54.152],
                                   [24.4927, 46.7352, 19.3509, 28.5914, 36.7449],
                                   [33.3941, 27.3918, 30.9353, 33.306, 27.0461],
                                   [-13.1568, -19.0059, -25.2635, -21.9022, -10.595]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f5

        # F11-F15: 50维度
        elif self.dim == 50:
            if self.prob_name == "F1":
                self.x_shift = np.array([[-43.0288, 70.8407, -43.1902, 35.5137, 49.2571, 6.7309,
                                          -31.8267, 10.5231, 54.7516, 47.4547, -15.9822, -58.556,
                                          12.5271, -54.959, 12.5443, -68.0263, -1.5204, 27.1992,
                                          70.4645, -1.9976, 19.4778, -18.6972, 54.7074, 45.891,
                                          1.1059, -76.231, 37.171, 63.8083, 24.9036, 56.645,
                                          21.5449, -16.8376, 74.5713, -33.3911, 56.8853, -45.4019,
                                          12.2432, 72.4271, -31.9359, -79.7986, 75.846, 55.8972,
                                          56.2525, 6.9595, 62.5999, 55.7957, 59.8969, -46.0762,
                                          -61.2451, 21.7185]])
                self.T = np.array([[-76.231, -78.3824, -60.0836, -65.4387, -90.4572],
                                   [37.171, 27.9376, 21.7284, 42.7057, 38.6151],
                                   [63.8083, 60.3516, 61.8176, 71.4459, 49.0497],
                                   [24.9036, 24.3683, 27.5275, 28.5457, 20.3032],
                                   [56.645, 53.9991, 45.1931, 56.6493, 65.3015],
                                   [21.5449, 16.9834, 26.7747, 3.7172, 19.3575],
                                   [-16.8376, -14.2269, -14.4773, -22.7987, -14.4654],
                                   [74.5713, 74.2963, 80.0208, 108.4417, 76.0704],
                                   [-33.3911, -43.023, -32.7041, -33.9852, -38.2591],
                                   [56.8853, 47.9509, 58.6457, 82.945, 41.3876],
                                   [-45.4019, -26.8774, -56.6721, -25.7065, -22.2223],
                                   [12.2432, 5.0709, 8.0227, -0.2025, 27.4882],
                                   [72.4271, 75.5349, 68.237, 80.0231, 65.4015],
                                   [-31.9359, -27.7543, -21.5223, -42.1046, -32.5595],
                                   [-79.7986, -84.9529, -86.6301, -82.9485, -79.5599],
                                   [75.846, 77.5003, 81.2406, 70.5975, 52.7937],
                                   [55.8972, 64.8054, 53.7556, 58.6102, 43.7699],
                                   [56.2525, 69.8101, 48.9739, 48.0522, 65.751],
                                   [6.9595, -24.9265, 21.9832, -8.2329, 14.1147],
                                   [62.5999, 75.1603, 59.6991, 48.0825, 63.2483],
                                   [55.7957, 66.2873, 50.9885, 70.7342, 57.1172],
                                   [59.8969, 72.3302, 56.1521, 60.6399, 61.9425],
                                   [-46.0762, -53.1465, -66.733, -33.9965, -51.7829],
                                   [-61.2451, -64.8116, -64.7768, -61.984, -61.0795],
                                   [21.7185, 21.129, 5.9121, 22.488, 35.3597]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f1

            elif self.prob_name == "F2":
                self.x_shift = np.array([[77.0666, -32.9016, -40.244, -78.2088, -10.7164, -52.7797,
                                          -55.995, 2.935, -35.9326, -24.9478, -9.6053, -76.9575,
                                          -12.6906, 50.4066, 12.6503, 45.1599, 53.5763, 15.8779,
                                          12.9312, 5.5481, -75.5946, -25.323, 64.3912, 68.1397,
                                          -75.0057, 34.4438, 18.17, -9.4799, -24.5676, 28.7979,
                                          -63.6044, -52.7973, -23.714, -71.6814, 31.2038, 36.009,
                                          -48.7627, 55.3539, 69.2254, -1.1358, -67.619, -77.8352,
                                          62.7575, 33.5628, 63.682, -5.8888, -54.2906, -48.2649,
                                          -44.5524, 0.5602]])
                self.T = np.array([[34.4438, 40.0134, 31.9421, 13.2089, 40.7778],
                                   [18.17, 22.2638, 28.5514, 33.6471, 41.8002],
                                   [-9.4799, 7.7258, 3.4416, -1.0876, 3.0162],
                                   [-24.5676, -14.3502, -18.1003, -14.2687, -24.2652],
                                   [28.7979, 40.6576, 39.9569, 14.714, 31.0407],
                                   [-63.6044, -66.3392, -56.0158, -75.4439, -74.4769],
                                   [-52.7973, -50.1803, -50.5987, -52.0407, -66.9335],
                                   [-23.714, -30.2862, -13.4269, -12.0912, -13.2247],
                                   [-71.6814, -80.4504, -74.1468, -88.9535, -77.373],
                                   [31.2038, 25.2644, 18.6164, 38.902, 42.65],
                                   [36.009, 35.6562, 37.455, 31.781, 36.3899],
                                   [-48.7627, -50.0634, -43.535, -44.434, -27.7206],
                                   [55.3539, 56.6805, 57.8984, 68.762, 53.8609],
                                   [69.2254, 76.2066, 68.4969, 53.4989, 74.3157],
                                   [-1.1358, 4.7003, 2.8575, 2.9152, 0.6341],
                                   [-67.619, -69.1429, -75.1107, -83.9894, -73.9739],
                                   [-77.8352, -78.1168, -84.0201, -69.632, -90.2173],
                                   [62.7575, 81.396, 53.8988, 65.782, 67.1557],
                                   [33.5628, 39.8969, 36.3304, 38.8745, 28.1104],
                                   [63.682, 59.1224, 67.6724, 63.1544, 60.6256],
                                   [-5.8888, -24.2175, -17.6429, -1.3088, -0.554],
                                   [-54.2906, -40.5351, -55.8705, -47.0743, -58.0414],
                                   [-48.2649, -49.4432, -50.4714, -58.1582, -39.1053],
                                   [-44.5524, -38.6248, -52.4938, -55.279, -36.2882],
                                   [0.5602, -7.343, -5.1049, -11.1241, 4.9232]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f2

            elif self.prob_name == "F3":
                self.x_shift = np.array([[21.3581, -34.7988, -48.2024, 5.2708, 52.5951, -57.111,
                                          70.6173, -73.0129, -63.0319, 23.4255, 65.1848, 9.7756,
                                          79.9119, -55.264, -6.7921, 26.7992, 64.02, -52.3316,
                                          60.9698, -52.9453, 12.2058, -67.3992, -47.4381, 4.898,
                                          -77.1715, 0.677, -39.0573, -68.9218, -8.7813, 60.3387,
                                          29.1911, 56.9715, 16.1028, 46.341, -29.8637, -10.1747,
                                          0.9009, 65.6017, 55.9991, 0.9459, -62.9375, 20.6265,
                                          12.8616, -38.6298, 6.2623, 48.3111, -2.7776, 74.7843,
                                          40.9011, -39.046]])

                self.T = np.array([[0.677, 4.2439, -14.3348, -2.1935, 2.8225],
                                   [-39.0573, -41.0163, -48.222, -34.6328, -43.6978],
                                   [-68.9218, -83.3037, -79.0681, -66.9423, -68.2008],
                                   [-8.7813, -20.3891, 19.7195, -14.1557, -14.3396],
                                   [60.3387, 63.7124, 63.9382, 63.1358, 45.5003],
                                   [29.1911, 26.3572, 24.9566, 19.8001, 33.6988],
                                   [56.9715, 67.4676, 59.6295, 61.0392, 53.1274],
                                   [16.1028, 15.6021, 21.3552, 24.1492, 15.9101],
                                   [46.341, 51.9029, 44.0415, 49.9335, 49.3361],
                                   [-29.8637, -29.5947, -22.8044, -35.329, -28.3595],
                                   [-10.1747, -4.861, -13.2413, -3.6554, -9.9668],
                                   [0.9009, 10.2315, -18.1587, -12.5843, 12.6243],
                                   [65.6017, 71.8017, 58.5299, 79.4843, 62.0589],
                                   [55.9991, 46.5067, 62.1489, 54.6877, 77.7314],
                                   [0.9459, 7.9803, 6.1928, -1.4781, 12.6058],
                                   [-62.9375, -55.5116, -55.1868, -56.9796, -66.0019],
                                   [20.6265, 16.0644, 17.3022, 23.0926, 3.692],
                                   [12.8616, 11.3397, 16.4233, 5.4268, 22.6829],
                                   [-38.6298, -27.7016, -36.3634, -44.3313, -47.122],
                                   [6.2623, -1.2012, 20.535, -8.0829, 15.1157],
                                   [48.3111, 67.2432, 54.3893, 40.9495, 46.5109],
                                   [-2.7776, 0.5098, -3.2313, 9.6477, 5.6439],
                                   [74.7843, 50.7893, 74.6405, 97.2279, 68.6691],
                                   [40.9011, 43.5626, 40.5884, 26.1296, 46.6742],
                                   [-39.046, -37.4981, -23.5071, -28.5136, -43.2813]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f3

            elif self.prob_name == "F4":
                self.x_shift = np.array([[-44.7459, -60.871, 5.6128, 16.8839, 55.8701, 1.0223,
                                          -24.6902, 10.1045, -46.1635, -51.6367, -30.2688, 78.4647,
                                          24.9505, 20.9986, 37.5911, -24.362, 50.1852, -34.9493,
                                          -11.9617, -32.9966, -77.0185, -47.6612, 2.2008, -65.8784,
                                          -52.132, 22.5557, 46.2536, 22.9362, 31.6661, -23.6212,
                                          30.4885, 46.689, -62.9374, -12.4109, -16.0204, -11.7627,
                                          74.9916, 76.6147, 31.8211, -23.4983, 79.7797, -53.688,
                                          -61.0718, -13.7119, -13.114, 74.0463, -4.4252, 58.3521,
                                          11.9745, -68.1321]])
                self.T = np.array([[22.5557, 41.6321, 27.3548, 30.7065, 18.7282],
                                   [46.2536, 55.0604, 47.9173, 44.5854, 41.6811],
                                   [22.9362, 33.1319, 11.3962, 26.7805, 12.1627],
                                   [31.6661, 38.7848, 1.3703, 36.2697, 21.2644],
                                   [-23.6212, -28.9686, -10.3636, -34.5237, -20.1622],
                                   [30.4885, 32.7563, 35.8975, 29.3221, 36.6328],
                                   [46.689, 48.3906, 45.0773, 45.7105, 60.4025],
                                   [-62.9374, -53.8223, -63.3685, -54.932, -62.153],
                                   [-12.4109, -12.8976, -4.841, -14.3732, -18.5134],
                                   [-16.0204, -31.3329, -17.6583, -17.3838, -2.9909],
                                   [-11.7627, -8.7493, -20.4464, -14.9079, -20.3899],
                                   [74.9916, 84.6758, 67.4054, 70.814, 57.745],
                                   [76.6147, 78.543, 69.5769, 99.3143, 85.3411],
                                   [31.8211, 52.813, 27.7016, 33.2542, 27.399],
                                   [-23.4983, -22.1509, -22.7033, -0.4958, -7.3071],
                                   [79.7797, 78.5805, 76.1528, 82.3443, 75.8703],
                                   [-53.688, -47.3586, -50.1599, -58.246, -50.3539],
                                   [-61.0718, -66.1843, -53.5219, -66.5575, -57.0088],
                                   [-13.7119, -6.3444, -16.2507, -1.096, -3.6727],
                                   [-13.114, -12.793, -5.9664, -19.5382, -19.8972],
                                   [74.0463, 81.3594, 71.1438, 80.4641, 94.42],
                                   [-4.4252, 2.8632, -24.8776, -14.1875, -12.2016],
                                   [58.3521, 60.8808, 47.4601, 70.2335, 62.8115],
                                   [11.9745, 28.8968, 2.8797, 11.6073, 7.8102],
                                   [-68.1321, -59.1678, -61.4943, -60.237, -68.0876]])

                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f4

            elif self.prob_name == "F5":
                self.x_shift = np.array([[44.0773, -39.1797, -44.166, -37.9102, -46.0301, 74.9386,
                                          36.1501, -12.295, -16.9456, 49.4559, -32.6329, 5.6998,
                                          -46.0868, 47.3142, 75.5664, -75.3266, -78.5506, 3.6234,
                                          -13.0037, -33.5193, -9.1332, 42.3564, -7.0175, -76.3318,
                                          75.8718, 42.0231, 16.1695, -22.2097, 28.2788, 1.3509,
                                          21.8213, 65.488, 30.1409, 31.0202, -73.9416, -14.8815,
                                          -47.1766, 37.117, -14.1003, 50.5048, -49.4264, 78.1375,
                                          40.5083, 6.4652, -21.5168, 4.4162, 31.2088, 21.0997,
                                          -58.6418, -78.507]])
                self.T = np.array([[42.0231, 35.1737, 28.1024, 42.5984, 35.3367],
                                   [16.1695, 22.5337, 13.504, 23.2691, 8.8752],
                                   [-22.2097, -45.022, -4.7113, -25.7122, -28.9579],
                                   [28.2788, 27.6983, 26.8522, 19.6075, 42.3518],
                                   [1.3509, 10.614, -5.6964, 3.1132, -19.2161],
                                   [21.8213, 17.8682, 20.9365, 21.8211, 15.3993],
                                   [65.488, 61.3377, 68.2773, 67.0436, 66.8365],
                                   [30.1409, 7.2689, 20.4235, 42.5332, 41.5341],
                                   [31.0202, 9.5749, 31.1954, 14.6897, 29.3831],
                                   [-73.9416, -74.622, -83.6204, -62.4362, -74.3343],
                                   [-14.8815, -23.1777, -22.9815, -4.4995, -10.0358],
                                   [-47.1766, -60.945, -54.1005, -50.033, -41.3555],
                                   [37.117, 35.4328, 41.4954, 52.1835, 38.2534],
                                   [-14.1003, -28.8248, -10.4855, -11.3978, -11.5233],
                                   [50.5048, 55.5375, 46.3197, 27.8673, 64.328],
                                   [-49.4264, -60.2289, -52.531, -61.0774, -56.098],
                                   [78.1375, 77.0122, 91.4892, 70.581, 93.6884],
                                   [40.5083, 26.1912, 43.158, 29.5171, 35.573],
                                   [6.4652, 19.4885, 1.638, -1.6075, -4.5114],
                                   [-21.5168, -26.2354, -27.9102, -15.2447, -9.4877],
                                   [4.4162, 7.3874, 2.7392, 11.996, 1.7837],
                                   [31.2088, 31.5358, 31.8352, 17.0965, 16.1361],
                                   [21.0997, 22.7531, 16.1743, 27.2555, 5.3971],
                                   [-58.6418, -66.301, -76.2324, -61.2812, -65.8844],
                                   [-78.507, -80.6636, -87.1408, -66.5553, -72.8899]])
                for i in range(self.dim - self.r):
                    self.N_lst.append(len(self.T[i]))
                self.F = f5

