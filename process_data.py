import numpy as np, re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file

def static_or_dynamic(X, users):
    feature_change_counts_per_user = np.zeros((len(users), X.shape[1]))

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
    X_chunk_per_user_static = [list() for i in range(len(users))]  # This will have 1 entry per user, where each entry is a full X_dataset that can be used for training
    X_chunk_per_user_dynamic = [list() for i in range(len(users))]  # This will have 1 entry per user, where each entry is a full X_dataset that can be used for training

    for index, row in enumerate(X):
        current_user = users[index]
        users_features_dynamic = dynamic_user_rows[current_user]
        users_features_static = static_user_rows[current_user]

        new_row_static = []
        new_row_dynamic = []
        
        for feature, value in enumerate(row):
            if feature in users_features_static:
                new_row_static.append(value)

            if feature in users_features_dynamic:
                new_row_dynamic.append(value)
        
        X_chunk_per_user_static[current_user].append(new_row_static)
        X_chunk_per_user_dynamic[current_user].append(new_row_dynamic)

    return (X_chunk_per_user_static, X_chunk_per_user_dynamic)

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

if __name__ == "__main__":
    X, y = parse_data("data/Bibtex/Bibtex_data.txt")
    np.random.seed(3)

    # Add user ID randonly
    num_unique_users = 2
    users = np.random.randint(num_unique_users, size=X.shape[0])
   
    X_chunk_per_user_static, X_chunk_per_user_dynamic = static_or_dynamic(X, users) # array with row for each user and colum for each feature. Each value is 0 if static and 1 if dynamic

    np.save('2_users_static.npy', np.array(X_chunk_per_user_static, dtype=object))
    np.save('2_users_dynamic.npy', np.array(X_chunk_per_user_dynamic, dtype=object))
