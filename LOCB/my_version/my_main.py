import argparse
from my_LOCB import LOCB, DA_LOCB, DA_LOCB_Static
import numpy as np
import sys 
from my_load_data import load_eedi_LOCB, load_eedi_DALOCB, load_eedi_DALOCB_static, load_ednet_DALOCB, load_ednet_LOCB
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
        user, context, rwd, question_id, user_correct = data.step()

        arm_select = model.recommend(user, context, t)
        r = rwd[arm_select]
        total_rwd = total_rwd + r
        reward_per_batch += r
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)

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

    reward_plot.plot(round, avg_rwd, label="DALOCB " + str(model.num_clusters) + " " + str(model.update_user_frequency) + " " + str(model.update_cluster_frequency))
    regret_plot.plot(round, avg_regret, label="DALOCB" + str(model.num_clusters) + " " + str(model.update_user_frequency) + " " + str(model.update_cluster_frequency))

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
        user, context, rwd, question_id, user_correct = data.step()

        arm_select = model.recommend(user, context, t)        
        r = rwd[arm_select]
        total_rwd = total_rwd + r
        reward_per_batch += r
        reg = np.max(rwd) - r
        summ+=reg
        regrets.append(summ)

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
        u, context, rwd = data.step()

        arm_select = model.recommend(u, context, t)
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
    parser.add_argument('--dataset', default='movie', type=str, help='eedi_DALOCB_dynamic, eedi_LOCB_dynamic, eedi_compare, ednet_compare, ednet_DALOCB_dynamic, ednet_LOCB_dynamic, eedi_DALOCB_static_dynamic, ednet_compare_cluster_size, ednet_compare_user_update_frequency, ednet_compare_cluster_update_frequency')
    args = parser.parse_args()
    data = args.dataset

    # Reward 
    _, reward_plot = plt.subplots()
    # Regret
    _, regret_plot = plt.subplots()

    np.random.seed(3)

    if data == "eedi_DALOCB_dynamic":
        b = load_eedi_DALOCB()
        model = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
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
        model_b = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
        
        c = load_eedi_DALOCB_static()
        model_c = DA_LOCB_Static(num_users = c.num_user, d = c.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
    
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
        model_b = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
        
        run_experiment_LOCB(data=a, model=model_a, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_b, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: LOCB vs. DA-LOCB on EdNet Dataset")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: LOCB vs. DA-LOCB on EdNet Dataset")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")

    elif data == "ednet_compare_cluster_size":    
        b = load_ednet_DALOCB()

        model_3 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
        model_5 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=5, update_cluster_frequency=1000, update_user_frequency=10)
        model_10 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=10, update_cluster_frequency=1000, update_user_frequency=10)
        model_15 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=15, update_cluster_frequency=1000, update_user_frequency=10)
        
        run_experiment_DALOCB(data=b, model=model_3, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_5, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_10, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_15, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: DA-LOCB on EdNet Dataset with Different Cluster Sizes")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: DA-LOCB on EdNet Dataset with Different Cluster Sizes")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")

    elif data == "ednet_compare_user_update_frequency":    
        b = load_ednet_DALOCB()

        model_5 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=5)
        model_10 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
        model_50 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=50)
        model_100 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=100)
        
        run_experiment_DALOCB(data=b, model=model_5, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_10, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_50, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_100, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: DA-LOCB on EdNet Dataset with Different User Update Frequencies")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: DA-LOCB on EdNet Dataset with Different User Update Frequencies")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")

    elif data == "ednet_compare_cluster_update_frequency":    
        b = load_ednet_DALOCB()

        model_500 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=500, update_user_frequency=10)
        model_1000 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
        model_2000 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=2000, update_user_frequency=10)
        model_5000 = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=5000, update_user_frequency=10)
        
        run_experiment_DALOCB(data=b, model=model_500, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_1000, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_2000, reward_plot=reward_plot, regret_plot=regret_plot)
        run_experiment_DALOCB(data=b, model=model_5000, reward_plot=reward_plot, regret_plot=regret_plot)

        reward_plot.set_title("Cumulative Mean Reward: DA-LOCB on EdNet Dataset with Different Cluster Update Frequencies")
        reward_plot.set_xlabel("Rounds")
        reward_plot.set_ylabel("Cumulative Mean Reward")

        regret_plot.set_title("Cumulative Mean Regret: DA-LOCB on EdNet Dataset with Different Cluster Update Frequencies")
        regret_plot.set_xlabel("Rounds")
        regret_plot.set_ylabel("Cumulative Mean Regret")

    elif data == "ednet_DALOCB_dynamic":
        b = load_ednet_DALOCB()

        model = DA_LOCB(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
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

        model = DA_LOCB_Static(num_users = b.num_user, d = b.dim, gamma = 0.2, delta = 0.1, detect_cluster = 0, num_clusters=3, update_cluster_frequency=1000, update_user_frequency=10)
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
