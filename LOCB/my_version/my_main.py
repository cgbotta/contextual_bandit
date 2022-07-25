# import os
import argparse
from cProfile import label
from my_LOCB import LOCB, DA_LOCB
import numpy as np
# from collections import defaultdict
import sys 
from my_load_data import load_movielen_new, load_yelp_new, load_eedi_LOCB, load_eedi_DALOCB
import time
import matplotlib.pyplot as plt

def run_experiment_DALOCB(data, model, reward_plot, regret_plot):
    regrets = []
    summ = 0
    reward_per_batch = 0
    rewards = []
    cumulative_reward = 0
    cumulative_reward_series = []
    print("Round; Regret; Regret/Round; Reward")
    avg_regret = []
    total_rwd = 0
    avg_rwd = []
    round = []
    for t in range(10000):
        # Basically, if I can get user, context, reward into this alg at this point,
        # it can do all the rest I believe
        # 
        # u is a number 0-k representing a single user. 
        # IMPORTANT: In this case, the users are actually clusters of similar users to get more data
        # context is a 10x10 matrix - 10 rows wach with 10 values: 
        # rwd is 10 element sparse array with a single 1 representing the reward for each arm (only 1 arms leads to positive reward)
        user, context, rwd, question_id, user_correct = data.step()

        arm_select = model.recommend(user, context, t)
        # arm_select = np.random.choice(range(10))
        
        r = rwd[arm_select]
        total_rwd = total_rwd + r
        reward_per_batch += r
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)
        # This is the bottleneck in eedi dataset
        # Pretty sure it is because the context is length 388 rather than 10
        # Maybe I can only use some of the most common subjects like I did users
        model.store_info(user = user, x = context[arm_select], r =r, t = t, question_id = question_id, user_correct = user_correct, question_subject_map = data.question_subject_map, subject_map=data.subjects)
        model.update(i = user, t =t)           
        if t % 50 == 0:
            print('{}: {:}, {:.4f}, {}'.format(t, summ, summ/(t+1), reward_per_batch))
            rewards.append(reward_per_batch)
            cumulative_reward_series.append(cumulative_reward)

            reward_per_batch = 0
            avg_regret.append(summ/(t+1))
            avg_rwd.append(total_rwd/(t+1))
            round.append(t)
  

    print("round:", t, "; ", "regret:", summ)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("personal:",model.personal)
    print(model.group_total)

    reward_plot.plot(round, avg_rwd, label="DALOCB")
    regret_plot.plot(round, avg_regret, label="DALOCB")

    np.save("./regret",  regrets)
    final_clusters = np.load("./results/clusters.npy", allow_pickle = True)

def run_experiment_LOCB(data, model, reward_plot, regret_plot):
    regrets = []
    summ = 0
    reward_per_batch = 0
    rewards = []
    cumulative_reward = 0
    cumulative_reward_series = []
    print("Round; Regret; Regret/Round; Reward")
    avg_regret = []
    total_rwd = 0
    avg_rwd = []
    round = []
    for t in range(10000):
        # IMPORTANT: In this case, the users are actually clusters of similar users to get more data
        # context is a 10x10 matrix - 10 rows wach with 10 values: 
        # rwd is 10 element sparse array with a single 1 representing the reward for each arm (only 1 arms leads to positive reward)
        u, context, rwd = data.step()

        arm_select = model.recommend(u, context, t)
        # arm_select = np.random.choice(range(10))

        r = rwd[arm_select]
        total_rwd = total_rwd + r
        reward_per_batch += r
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)
        model.store_info(i = u, x = context[arm_select], y =r, t = t)
        model.update(i = u, t =t)    
        if t % 50 == 0:
            print('{}: {:}, {:.4f}, {}'.format(t, summ, summ/(t+1), reward_per_batch))
            rewards.append(reward_per_batch)
            cumulative_reward_series.append(cumulative_reward)

            reward_per_batch = 0
            avg_regret.append(summ/(t+1))
            avg_rwd.append(total_rwd/(t+1))
            round.append(t)
    print("round:", t, "; ", "regret:", summ)

    reward_plot.plot(round, avg_rwd, label="LOCB")
    regret_plot.plot(round, avg_regret, label="LOCB")
    np.save("./regret",  regrets)


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Meta-Ban')
    parser.add_argument('--dataset', default='movie', type=str, help='yelp, movie, eedi_DALOCB, eedi_LOCB, eedi_compare')
    args = parser.parse_args()
    data = args.dataset

    # Reward 
    _, reward_plot = plt.subplots()
    # Regret
    _, regret_plot = plt.subplots()

    np.random.seed(3)

    
    if data == "yelp":
        b = load_yelp_new()
        model = LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)

        run_experiment_LOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)
    elif data == "movie":
        b = load_movielen_new()
        model = LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)

        run_experiment_LOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)
    elif data == "eedi_DALOCB":
        b = load_eedi_DALOCB()

        model = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
        run_experiment_DALOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)
    elif data == "eedi_LOCB":
        b = load_eedi_LOCB()
        print("Finished load_eedi_new() --- %s seconds ---" % (time.time() - start_time))

        model = LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 8, delta = 0.1,  detect_cluster = 0)
        run_experiment_LOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)
    elif data == "eedi_compare":
        a = load_eedi_LOCB()
        model_a = LOCB(num_users = a.num_user, d = a.dim, gamma = 0.2, num_seeds = 8, delta = 0.1,  detect_cluster = 0)
    
        b = load_eedi_DALOCB()
        model_b = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)

        run_experiment_LOCB(data=a, model=model_a, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_b, reward_plot=reward_plot, regret_plot=regret_plot)


    else:
        print("dataset is not defined. --help")
        sys.exit()
    
    reward_plot.legend()
    reward_plot.set_title("Cumulative Mean Reward: LOCB vs. DA-LOCB on Eedi Dataset")
    reward_plot.set_xlabel("Rounds")
    reward_plot.set_ylabel("Cumulative Mean Reward")

    
    regret_plot.legend()
    regret_plot.set_title("Cumulative Mean Regret: LOCB vs. DA-LOCB on Eedi Dataset")
    regret_plot.set_xlabel("Rounds")
    regret_plot.set_ylabel("Cumulative Mean Regret")

    plt.show()
