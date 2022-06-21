import os
import logging
import sys
import numpy as np


def one_hot(in_list, goal_size):
    # in shape: [1, 1] out shape: [1, goal_size]
    in_list = np.array(in_list)
    out_list = np.zeros((in_list.size, goal_size))
    out_list[np.arange(in_list.size), in_list] = 1
    return out_list

def make_train_data_v1(reward, done, value, gamma, num_step, num_worker, use_gae, lam):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = 0
        for t in range(num_step - 1, -1, -1):
            delta = reward[t] + gamma * value[t + 1] * (1 - done[t]) - value[t]
            gae = delta + gamma * lam * (1 - done[t]) * gae

            discounted_return[:, t] = gae + value[t]

            # For Actor
        adv = discounted_return - value[:-1]

    else:
        running_add = value[-1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[t] + gamma * running_add * (1 - done[t])
            discounted_return[t] = running_add

        # For Actor
        adv = discounted_return - value[:-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


def make_train_data(reward, done, value, gamma, num_step, num_worker, use_gae):
    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * lam * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


def get_logger(log_directory, clear_prev_log=False, logfile_name='info.log'):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_path = os.path.join(log_directory, logfile_name)
    if os.path.exists(log_path) and clear_prev_log:
        os.remove(log_path)

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s',
                                  datefmt='%a, %d %b %Y %H:%M:%S')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    setattr(logger, 'log_directory', log_directory)
    return logger
