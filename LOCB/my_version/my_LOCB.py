import numpy as np
from student_model import Student
import pandas as pd

class Cluster:
    def __init__(self, users, S, b, N):
        self.users = set(users)
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)

class LOCB:
    def __init__(self, num_users, d, gamma, num_seeds, delta, detect_cluster):
        self.S = {i:np.eye(d) for i in range(num_users)}
        self.b = {i:np.zeros(d) for i in range(num_users)}
        self.Sinv = {i:np.eye(d) for i in range(num_users)}
        self.theta = {i:np.zeros(d) for i in range(num_users)}
        self.users = range(num_users)

        self.seeds = np.random.choice(self.users, num_seeds)
        self.seed_state = {}
        for seed in self.seeds:
            self.seed_state[seed] = 0
        self.clusters = {}

        for seed in self.seeds: 
            self.clusters[seed] = Cluster(users=self.users, S=np.eye(d), b=np.zeros(d), N=1)
            
        self.N = np.zeros(num_users)
        self.gamma = gamma
        self.results = []
        self.fin = 0
        self.cluster_inds = {i:[] for i in range(num_users)}
        for i in self.users:
            for seed in self.seeds:
                if i in self.clusters[seed].users:
                    self.cluster_inds[i].append(seed)
                    
        self.d = d
        self.n = num_users
        self.selected_cluster = 0 
        self.delta = delta
        self.if_d = detect_cluster
        
    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta
    
    def recommend(self, u, items, t):
        cls = self.cluster_inds[u]
        if (len(cls)>0) and (t <40000):
            res = []
            for c in cls:
                cluster = self.clusters[c]
                res_sin = self._select_item_ucb(cluster.S,cluster.Sinv, cluster.theta, items, cluster.N, t)
                res.append(res_sin)
            best_cluster = max(res)
            return best_cluster[1]
        else:
            no_cluster = self._select_item_ucb(self.S[u], self.Sinv[u], self.theta[u], items, self.N[u], t)
            return no_cluster[1]
  
    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        ucbs = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1)
        res = max(ucbs)
        it = np.argmax(ucbs)
        return (res, it)
        
    def store_info(self, i, x, y, t):
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1
        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])
        
        for c in self.cluster_inds[i]:
            self.clusters[c].S += np.outer(x, x)
            self.clusters[c].b += y * x
            self.clusters[c].N += 1
            self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
            self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)
    
    def update(self, i, t):
        def _factT(m):
            if self.if_d:
                delta = self.delta / self.n
                nu = np.sqrt(2*self.d*np.log(1 + t) + 2*np.log(2/delta)) +1
                de = np.sqrt(1+m/4)*np.power(self.n, 1/3)
                return nu/de
            else:
                return np.sqrt((1 + np.log(1 + m)) / (1 + m))
        
        if not self.fin:
            for seed in self.seeds:
                if not self.seed_state[seed]:
                    if i in self.clusters[seed].users:
                        diff = self.theta[i] - self.theta[seed]
                        if np.linalg.norm(diff) > _factT(self.N[i]) + _factT(self.N[seed]):
                            self.clusters[seed].users.remove(i)
                            self.cluster_inds[i].remove(seed)                            
                            self.clusters[seed].S = self.clusters[seed].S - self.S[i] + np.eye(self.d)
                            self.clusters[seed].b = self.clusters[seed].b - self.b[i]
                            self.clusters[seed].N = self.clusters[seed].N - self.N[i]
                            
                    else:
                        diff = self.theta[i] - self.theta[seed]
                        if np.linalg.norm(diff) < _factT(self.N[i]) + _factT(self.N[seed]):
                            self.clusters[seed].users.add(i)
                            self.cluster_inds[i].append(seed)
                            self.clusters[seed].S = self.clusters[seed].S + self.S[i] - np.eye(self.d)
                            self.clusters[seed].b = self.clusters[seed].b + self.b[i]
                            self.clusters[seed].N = self.clusters[seed].N + self.N[i]
                
                    if self.if_d: thre = self.gamma 
                    else: thre = self.gamma/4
                    if _factT(self.N[seed]) <= thre:
                        self.seed_state[seed] = 1
                        self.results.append({seed:list(self.clusters[seed].users)}) 
            finished = 1
            for i in self.seed_state.values():
                if i ==0:
                    finished =0
                    
            if finished: 
                if self.if_d:
                    np.save('./results/clusters', self.results)
                    print('Clustering finished! Round:', t)
                    print(len(self.results))
                    print(self.results)
                    self.stop = 1
                self.fin = 1
                  
class DA_Cluster:
    def __init__(self, users, S, b, N):
        self.users = set(users)
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)

class DA_LOCB:
    def __init__(self, num_users, d, gamma, delta, detect_cluster, num_clusters, update_cluster_frequency, update_user_frequency):
        self.S = {i:np.eye(d) for i in range(num_users)}
        self.b = {i:np.zeros(d) for i in range(num_users)}
        self.Sinv = {i:np.eye(d) for i in range(num_users)}
        self.theta = {i:np.zeros(d) for i in range(num_users)}
        self.users = range(num_users)

        self.DA_Students = {}
        for user in self.users:
            self.DA_Students[user] = Student(user) 
        
        self.clusters = {}
        self.num_clusters = num_clusters
        self.N = np.zeros(num_users)
        self.gamma = gamma
        self.results = []
        self.fin = 0
        self.groups = [0] * num_users          
        self.d = d
        self.n = num_users
        self.selected_cluster = 0 
        self.delta = delta
        self.if_d = detect_cluster
        self.update_cluster_frequency = update_cluster_frequency
        self.update_user_frequency = update_user_frequency
        
    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta
    
    def recommend(self, u, items, t):
        if not self.clusters:
            no_cluster = self._select_item_ucb(self.S[u], self.Sinv[u], self.theta[u], items, self.N[u], t)
            return no_cluster[1]
        else:
            c = self.groups[u]
            no_cluster = self._select_item_ucb(self.S[u], self.Sinv[u], self.theta[u], items, self.N[u], t)
            cluster = self.clusters[c]
            cluster_response = self._select_item_ucb(cluster.S,cluster.Sinv, cluster.theta, items, cluster.N, t)
            maximium = max(no_cluster, cluster_response)
            return maximium[1]  
        
    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        ucbs = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1)
        res = max(ucbs)
        it = np.argmax(ucbs)
        return (res, it)

    def store_info(self, user, x, r, t, question_id, user_correct, question_subject_map, subject_map):
        self.S[user] += np.outer(x, x)
        self.b[user] += r * x
        self.N[user] += 1
        self.Sinv[user], self.theta[user] = self._update_inverse(self.S[user], self.b[user], self.Sinv[user], x, self.N[user])
        
        if self.clusters:
            c = self.groups[user]
            self.clusters[c].S += np.outer(x, x)
            self.clusters[c].b += r * x
            self.clusters[c].N += 1
            if np.linalg.det(self.clusters[c].S) != 0:
                self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
            self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)
        
        # Update the student subject accuracies
        self.DA_Students[user].add_question_answered_percentage(question_id, user_correct, question_subject_map[question_id])
        self.DA_Students[user].get_subject_accuracies_array(subject_map, self.update_user_frequency)

    def update(self, i, t):
        if t % self.update_cluster_frequency == 0 and t > 0:
            # Get all the dynamics for all the users
            dynamics = []

            for user in self.users:
                avg_diff = self.DA_Students[user].average_diff
                temp_dict = (user,avg_diff)
                dynamics.append(temp_dict)

            df = pd.DataFrame(dynamics)
            df_sorted = df.sort_values(by=[1])

            user_list = df_sorted.iloc[:, 0].tolist()
            groups = np.array_split(user_list, self.num_clusters)

            new_groups = [0] * len(user_list)

            for user in self.users:
                for index, sub_list in enumerate(groups):
                    if user in sub_list:
                        new_groups[user] = index            
            
            if t == self.update_cluster_frequency:
                # Cluster for the first time
                for cluster in range(self.num_clusters):
                    users_for_cluster = groups[cluster]
                    self.clusters[cluster] = Cluster(users=users_for_cluster, S=np.eye(self.d), b=np.zeros(self.d), N=1)
                self.groups = new_groups

            else:
                # Otherwise, after clusters exist, just update the clusters as needed
                for user in range(len(new_groups)):
                    new_group = new_groups[user]
                    old_group = self.groups[user]
                    if new_group != old_group:
                        self.clusters[old_group].users.remove(user)
                        self.clusters[old_group].S = self.clusters[old_group].S - self.S[user] + np.eye(self.d)
                        self.clusters[old_group].b = self.clusters[old_group].b - self.b[user]
                        self.clusters[old_group].N = self.clusters[old_group].N - self.N[user]
                           
                        self.clusters[new_group].users.add(user)
                        self.clusters[new_group].S = self.clusters[new_group].S + self.S[user] - np.eye(self.d)
                        self.clusters[new_group].b = self.clusters[new_group].b + self.b[user]
                        self.clusters[new_group].N = self.clusters[new_group].N + self.N[user]
                self.groups = new_groups
                                                  
class DA_LOCB_Static:
    def __init__(self, num_users, d, gamma, delta, detect_cluster, num_clusters, update_cluster_frequency, update_user_frequency):
        self.S = {i:np.eye(d) for i in range(num_users)}
        self.b = {i:np.zeros(d) for i in range(num_users)}
        self.Sinv = {i:np.eye(d) for i in range(num_users)}
        self.theta = {i:np.zeros(d) for i in range(num_users)}
        self.users = range(num_users)

        self.DA_Students = {}
        for user in self.users:
            self.DA_Students[user] = Student(user) 

        self.clusters = {}
        self.num_clusters = num_clusters
        self.N = np.zeros(num_users)
        self.gamma = gamma
        self.results = []
        self.fin = 0
        self.groups = [0] * num_users    
        self.d = d
        self.n = num_users
        self.selected_cluster = 0 
        self.delta = delta
        self.if_d = detect_cluster
        self.update_cluster_frequency = update_cluster_frequency
        self.update_user_frequency = update_user_frequency
        
    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1))

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta
    
    def recommend(self, u, items, t):
        if not self.clusters:
            no_cluster = self._select_item_ucb(self.S[u], self.Sinv[u], self.theta[u], items, self.N[u], t)
            return no_cluster[1]
        else:
            c = self.groups[u]
            no_cluster = self._select_item_ucb(self.S[u], self.Sinv[u], self.theta[u], items, self.N[u], t)
            cluster = self.clusters[c]
            cluster_response = self._select_item_ucb(cluster.S,cluster.Sinv, cluster.theta, items, cluster.N, t)
            maximium = max(no_cluster, cluster_response)
            return maximium[1]
  
    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        ucbs = np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis = 1)
        res = max(ucbs)
        it = np.argmax(ucbs)
        return (res, it)
        
    def store_info(self, user, x, r, t, question_id, user_correct, question_subject_map, subject_map, user_metadata_map):
        self.S[user] += np.outer(x, x)
        self.b[user] += r * x
        self.N[user] += 1
        self.Sinv[user], self.theta[user] = self._update_inverse(self.S[user], self.b[user], self.Sinv[user], x, self.N[user])
        
        if self.clusters:
            c = self.groups[user]
            self.clusters[c].S += np.outer(x, x)
            self.clusters[c].b += r * x
            self.clusters[c].N += 1
            if np.linalg.det(self.clusters[c].S) != 0:
                self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
            self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)
        
        # Update the student subject accuracies
        self.DA_Students[user].add_question_answered_percentage(question_id, user_correct, question_subject_map[question_id])
        self.DA_Students[user].get_subject_accuracies_array_static(subject_map, user_metadata_map, self.update_user_frequency)


    def update(self, i, t):
        if t % self.update_cluster_frequency == 0 and t > 0:
            # Get all the dynamics for all the users
            dynamics = []

            for user in self.users:
                avg_diff = self.DA_Students[user].average_diff
                temp_dict = (user,avg_diff)
                dynamics.append(temp_dict)

            df = pd.DataFrame(dynamics)
            df_sorted = df.sort_values(by=[1])

            user_list = df_sorted.iloc[:, 0].tolist()
            groups = np.array_split(user_list, self.num_clusters)

            new_groups = [0] * len(user_list)

            for user in self.users:
                for index, sub_list in enumerate(groups):
                    if user in sub_list:
                        new_groups[user] = index                
            
            if t == self.update_cluster_frequency:
                # Cluster for the first time
                for cluster in range(self.num_clusters):
                    users_for_cluster = groups[cluster]
                    self.clusters[cluster] = Cluster(users=users_for_cluster, S=np.eye(self.d), b=np.zeros(self.d), N=1)
                self.groups = new_groups

            else:
                # Otherwise, after clusters exist, just update the clusters as needed
                for user in range(len(new_groups)):
                    new_group = new_groups[user]
                    old_group = self.groups[user]
                    if new_group != old_group:
                        self.clusters[old_group].users.remove(user)
                        self.clusters[old_group].S = self.clusters[old_group].S - self.S[user] + np.eye(self.d)
                        self.clusters[old_group].b = self.clusters[old_group].b - self.b[user]
                        self.clusters[old_group].N = self.clusters[old_group].N - self.N[user]
                           
                        self.clusters[new_group].users.add(user)
                        self.clusters[new_group].S = self.clusters[new_group].S + self.S[user] - np.eye(self.d)
                        self.clusters[new_group].b = self.clusters[new_group].b + self.b[user]
                        self.clusters[new_group].N = self.clusters[new_group].N + self.N[user]
                self.groups = new_groups
