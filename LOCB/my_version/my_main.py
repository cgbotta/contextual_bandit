# import os
import argparse
from cProfile import label
from my_LOCB import LOCB, DA_LOCB, DA_LOCB_Static
import numpy as np
# from collections import defaultdict
import sys 
from my_load_data import load_movielen_new, load_yelp_new, load_eedi_LOCB, load_eedi_DALOCB, load_eedi_DALOCB_static, load_ednet_DALOCB, load_ednet_LOCB
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
    for t in range(20000):
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
    print("personal:",model.personal)
    print(model.group_total)

    reward_plot.plot(round, avg_rwd, label="DALOCB")
    regret_plot.plot(round, avg_regret, label="DALOCB")

    np.save("./regret",  regrets)

def run_experiment_DALOCB_static(data, model, reward_plot, regret_plot):
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
    for t in range(20000):
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
        model.store_info(user = user, x = context[arm_select], r =r, t = t, question_id = question_id, user_correct = user_correct, question_subject_map = data.question_subject_map, subject_map=data.subjects, user_metadata_map = data.user_metadata_map)
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
    print("personal:",model.personal)
    print(model.group_total)

    reward_plot.plot(round, avg_rwd, label="DALOCB + static")
    regret_plot.plot(round, avg_regret, label="DALOCB + static")

    np.save("./regret",  regrets)

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
    for t in range(20000):
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
    parser = argparse.ArgumentParser(description='Meta-Ban')
    parser.add_argument('--dataset', default='movie', type=str, help='yelp, movie, eedi_DALOCB_dynamic, eedi_LOCB_dynamic, eedi_compare, ednet_compare, ednet_DALOCB_dynamic, ednet_LOCB_dynamic, eedi_DALOCB_static_dynamic')
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
    elif data == "eedi_DALOCB_dynamic":
        b = load_eedi_DALOCB()
        model = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
        run_experiment_DALOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: DA-LOCB on Eedi Dataset (1_2)")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: DA-LOCB on Eedi Dataset (1_2)")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")
    elif data == "eedi_LOCB_dynamic":
        b = load_eedi_LOCB()
        model = LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)
        run_experiment_LOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: LOCB on Eedi Dataset (1_2)")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: LOCB on Eedi Dataset (1_2)")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")
    elif data == "eedi_compare":
        a = load_eedi_LOCB()
        model_a = LOCB(num_users = a.num_user, d = a.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)
    
        b = load_eedi_DALOCB()
        model_b = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
        
        c = load_eedi_DALOCB_static()
        model_c = DA_LOCB_Static(num_users = c.num_user, d = c.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
    
        run_experiment_LOCB(data=a, model=model_a, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_b, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB_static(data=c, model=model_c, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: LOCB vs. DA-LOCB on Eedi Dataset (1_2)")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: LOCB vs. DA-LOCB on Eedi Dataset (1_2)")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")

    elif data == "ednet_compare":
        a = load_ednet_LOCB()
        model_a = LOCB(num_users = a.num_user, d = a.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)
    
        b = load_ednet_DALOCB()
        model_b = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
        
    
        run_experiment_LOCB(data=a, model=model_a, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_b, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: LOCB vs. DA-LOCB on EdNet Dataset")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: LOCB vs. DA-LOCB on EdNet Dataset")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")

    elif data == "ednet_DALOCB_dynamic":
        b = load_ednet_DALOCB()

        model = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
        run_experiment_DALOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: DA-LOCB on EdNet Dataset")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: DA-LOCB on EdNet Dataset")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")
    elif data == "ednet_LOCB_dynamic":
        b = load_ednet_LOCB()

        model = LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, num_seeds = 20, delta = 0.1,  detect_cluster = 0)
        run_experiment_LOCB(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: LOCB on EdNet Dataset")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: LOCB on EdNet Dataset")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")
    elif data == "eedi_DALOCB_static_dynamic":
        b = load_eedi_DALOCB_static()

        model = DA_LOCB_Static(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3)
        run_experiment_DALOCB_static(data=b, model=model, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: DA-LOCB + Static on Eedi Dataset (1_2)")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: DA-LOCB + Static on Eedi Dataset (1_2)")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")
    else:
        print("dataset is not defined. Run with --help for dataset options")
        sys.exit()
    
    reward_plot.legend()
    regret_plot.legend()
    plt.show()
