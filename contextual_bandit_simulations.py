import pandas as pd, numpy as np, re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from contextualbandits.online import BootstrappedUCB, BootstrappedTS, LogisticUCB, \
            SeparateClassifiers, EpsilonGreedy, AdaptiveGreedy, ExploreFirst, \
            ActiveExplorer, SoftmaxExplorer
from copy import deepcopy
import matplotlib.pyplot as plt

def parse_data(filename):
    with open(filename, "rb") as f:
        infoline = f.readline()
        infoline = re.sub(r"^b'", "", str(infoline))
        n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
        features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    features = np.array(features.todense())
    features = np.ascontiguousarray(features)
    return features, labels

def main(X, y):
    nchoices = y.shape[1]
    base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True)
    beta_prior = ((3./nchoices, 4), 2) # until there are at least 2 observations of each class, will use this prior
    ### Important!!! the default values for beta_prior will be changed in version 0.3

    ## The base algorithm is embedded in different metaheuristics
    epsilon_greedy = EpsilonGreedy(deepcopy(base_algorithm), nchoices = nchoices,
                                beta_prior = beta_prior, random_state = 4444)
    adaptive_greedy_thr = AdaptiveGreedy(deepcopy(base_algorithm), nchoices=nchoices,
                                        decay_type='threshold',
                                        beta_prior = beta_prior, random_state = 6666)

    models = [epsilon_greedy, adaptive_greedy_thr]


    # These lists will keep track of the rewards obtained by each policy
    rewards_egr, rewards_agr = [list() for i in range(len(models))]

    lst_rewards = [rewards_egr, rewards_agr]

    # batch size - algorithms will be refit after N rounds
    batch_size = 50

    # initial seed - all policies start with the same small random selection of actions/rewards
    first_batch = X[:batch_size, :]
    np.random.seed(1)
    action_chosen = np.random.randint(nchoices, size=batch_size)
    rewards_received = y[np.arange(batch_size), action_chosen]

    # fitting models for the first time
    for model in models:
        model.fit(X=first_batch, a=action_chosen, r=rewards_received)
        
    # these lists will keep track of which actions does each policy choose
    lst_a_egr, lst_a_agr = [action_chosen.copy() for i in range(len(models))]

    lst_actions = [lst_a_egr, lst_a_agr]

    # rounds are simulated from the full dataset
    def simulate_rounds(model, rewards, actions_hist, X_global, y_global, batch_st, batch_end):
        np.random.seed(batch_st)
        
        ## choosing actions for this batch
        actions_this_batch = model.predict(X_global[batch_st:batch_end, :]).astype('uint8')
        
        # keeping track of the sum of rewards received
        rewards.append(y_global[np.arange(batch_st, batch_end), actions_this_batch].sum())
        
        # adding this batch to the history of selected actions
        new_actions_hist = np.append(actions_hist, actions_this_batch)
        
        # now refitting the algorithms after observing these new rewards
        np.random.seed(batch_st)
        model.fit(X_global[:batch_end, :], new_actions_hist, y_global[np.arange(batch_end), new_actions_hist],
                warm_start = True)
        
        return new_actions_hist

    # now running all the simulation
    for i in range(int(np.floor(X.shape[0] / batch_size))):
        batch_st = (i + 1) * batch_size
        batch_end = (i + 2) * batch_size
        batch_end = np.min([batch_end, X.shape[0]])
        
        for model in range(len(models)): 
            lst_actions[model] = simulate_rounds(models[model],lst_rewards[model],lst_actions[model],X, y,batch_st, batch_end)

    lwd = 5
    cmap = plt.get_cmap('tab20')
    colors=plt.cm.tab20(np.linspace(0, 1, 20))

    ax = plt.subplot()
    plt.plot(get_mean_reward(rewards_agr, batch_size), label="Adaptive Greedy (decaying threshold)",linewidth=lwd,color=colors[10])
    plt.plot(np.repeat(y.mean(axis=0).max(),len(rewards_egr)), label="Overall Best Arm (no context)",linewidth=lwd,color=colors[1],ls='dashed')

    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks([i*20 for i in range(8)], [i*1000 for i in range(8)])

    plt.xlabel(f'Rounds (models were updated every {batch_size} rounds)', size=20)
    plt.ylabel('Cumulative Mean Reward', size=20)
    plt.title('Comparison of Online Contextual Bandit Policies\n(Base Algorithm is Logistic Regression)\n\nBibtext Dataset\n(159 categories, 1836 attributes)',size=20)
    plt.grid()
    plt.show()

def get_mean_reward(reward_lst, batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew

def static_or_dynamic(X, y, users, type):
    feature_change_counts_per_user = np.zeros((len(users), X.shape[1]))

    new_X = np.array(0)
    # Count how often features change, for each user
    last_context_per_user = np.zeros((len(users), X.shape[1]))
    for index, current_context in enumerate(X):
        user = users[index]
        previous_context = last_context_per_user[user]

        changed_indices = np.where(current_context != previous_context)
        
        # Update counts for features that changed
        for index_that_changed in changed_indices[0]:
            feature_change_counts_per_user[user][index_that_changed] = feature_change_counts_per_user[user][index_that_changed] + 1

        last_context_per_user[user] = current_context

    dynamic_user_rows = []
    static_user_rows = []
    # Figure out which feature change the most
    for user_counts in feature_change_counts_per_user:
        mean = np.mean(user_counts)

        dynamic_feature_indices = np.where(user_counts >= mean)[0]
        static_feature_indices = np.where (user_counts < mean)[0]


        # # Get array of 50 lists, each one is the proper X row for for that user
        # x_dynamic = np.zeros(X.shape[1])
        # x_static = np.zeros(X.shape[1])
        # for feature in dynamic_feature_indices:
        #     x_dynamic[feature] = 1
        # for feature in static_feature_indices:
        #     x_static[feature] = 1

        dynamic_user_rows.append(dynamic_feature_indices)
        static_user_rows.append(static_feature_indices)


    # I think I need to run the training process on a per user basis???? Could look into how to train a model with different features inputted for different users like this??
    X_chunk_per_user = [list() for i in range(len(users))]  # This will have 1 entry per user, where each entry is a full X_dataset that can be used for training

    for index, row in enumerate(X):
        current_user = users[index]
        if type == "dynamic":
            users_features = dynamic_user_rows[current_user]
        elif type == "static":
            users_features = static_user_rows[current_user]

        new_row = []
        for feature, value in enumerate(row):
            if feature in users_features:
                new_row.append(value)
        X_chunk_per_user[current_user].append(new_row)

    return X_chunk_per_user

def generate_y(y, users):
    y_chunk_per_user = [list() for i in range(len(users))]  # This will have 1 entry per user, where each entry is a full X_dataset that can be used for training

    for index, row in enumerate(y):
        current_user = users[index]
        # # users_dynamic_features = dynamic_user_rows[current_user]
        # new_row = []
        # for arm_label, value in enumerate(row):
        #     new_row.append(value)
        y_chunk_per_user[current_user].append(row)

    return y_chunk_per_user        
if __name__ == "__main__":
    X, y = parse_data("data/Bibtex/Bibtex_data.txt")
    np.random.seed(0)

    print(X.shape)
    print(y.shape)

    
    # Add user ID randonly
    num_unique_users = 1
    users = np.random.randint(num_unique_users, size=X.shape[0])
   
    X_chunk_per_user = static_or_dynamic(X, y, users, "static") # array with row for each user and colum for each feature. Each value is 0 if static and 1 if dynamic
    # Maybe save and load this data so I don't have to wait for it to calculate each run

    y_chunk_per_user = generate_y(y, users)
    # Train for each user
    for user in range(1):
        # main(X, y)
        main(np.array(X_chunk_per_user[user]), np.array(y_chunk_per_user[user]))

    



