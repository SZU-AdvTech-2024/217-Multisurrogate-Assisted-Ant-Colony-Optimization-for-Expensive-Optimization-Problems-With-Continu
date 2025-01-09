import numpy as np
import time
import os
from Benchmarks.MVOP_type3 import MVOPT3
from My_MiACO import MiACO

def run_MiSACO_benchmark(fun_name, n_c, n_d, maxFEs, trials):

    save_y_best = np.zeros(trials)
    save_c_reslut = np.zeros((maxFEs, trials))
    start_time = time.time()
    for i in range(trials):
        print("Running trial: ", i)
        problem = MVOPT3(fun_name, n_c, n_d)
        opt = MiACO(maxFEs=maxFEs, popsize=100, dim=problem.dim, clb=[problem.bounds[0]] * problem.r,
                      cub=[problem.bounds[1]] * problem.r, N_lst=problem.N_lst, v_dv=problem.v_dv, prob=problem.F,
                      r=problem.r)
        x_best, y_best, y_lst, database, c_result = opt.run()
        save_y_best[i] = y_best
        save_c_reslut[:, i] = c_result
    end_time = time.time()
    os.makedirs('./result', exist_ok=True)
    mean_value = np.mean(save_y_best)  #算平均值
    std_value = np.std(save_y_best)  #算标准差
    time_cost = end_time - start_time
    last_value = [mean_value, std_value, time_cost]
    # cov_curve = np.mean(save_c_reslut, axis=1)
    cov_curve = np.zeros((maxFEs, trials + 1))
    cov_curve[:, 0:trials] = save_c_reslut
    cov_curve[:, trials] = np.mean(save_c_reslut, axis=1)
    np.savetxt('./result/%s.txt' % fun_name, last_value)  # 均值与方差
    np.savetxt('./result/%s_Convergence curve.txt' % fun_name, cov_curve)


if __name__ == "__main__":
    total = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
             'F11', 'F12', 'F13', 'F14', 'F15',
             'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23', 'F24',
             'F25', 'F26', 'F27', 'F28', 'F29', 'F30']
    maxFEs = 600
    trials = 20

    for i in range(0, 5):
        fun_name = total[i]

        if total[i] in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]:
            n_c = 8
            n_d = 2
        elif total[i] in ["F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20"]:
            n_c = 2
            n_d = 8
        elif total[i] in ["F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28", "F29", "F30"]:
            n_c = 5
            n_d = 5

        run_MiSACO_benchmark(fun_name, n_c, n_d, maxFEs, trials)
