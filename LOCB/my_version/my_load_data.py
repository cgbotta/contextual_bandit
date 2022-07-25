from re import sub
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from statistics import mean
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
import pandas as pd
from student_model import Student
import matplotlib.pyplot as plt
import ast
import time
import random
from scipy.stats import norm
import pickle

class load_movielen_new:
    def __init__(self):
        # Fetch data
        self.m = np.load("../movie_2000users_10000items_entry.npy")
        self.U = np.load("../movie_2000users_10000items_features.npy")
        self.I = np.load("../movie_10000items_2000users_features.npy")

        kmeans = KMeans(n_clusters=50, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        print(self.groups)
        self.n_arm = 10
        self.dim = 10
        self.num_user = 50
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in self.m:
            if i[2] ==1:
                self.pos_index[self.groups[i[0]]].append((i[0], i[1]))
            else:
                self.neg_index[self.groups[i[0]]].append((i[0], i[1]))   


    def step(self):    
        u = np.random.choice(range(2000))
        g = self.groups[u]
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)
        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[ind[1]])
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        return g, contexts, rwd
    
class load_yelp_new:
    def __init__(self):
        # Fetch data
        # This appears to be a list of >100,000 tuples where each tuple is (user, restaurant, if they rated it over 3 stars)
        self.m = np.load("../yelp_2000users_10000items_entry.npy")
        # This is 2000 rows (1 for each user) with 10 columns (user features, perhaps)
        self.U = np.load("../yelp_2000users_10000items_features.npy")
        # This is 10,000 rows (1 for each restaurant, perhaps) with 10 columns (restarant features, perhaps)
        self.I = np.load("../yelp_10000items_2000users_features.npy")

        # Cluster based on the user features 
        kmeans = KMeans(n_clusters=50, random_state=0).fit(self.U)
        self.groups = kmeans.labels_
        self.n_arm = 10
        self.dim = 10
        self.num_user = 50

        # This chunk is crazy. It creates a dictionary of length num_users where the key
        # is the user_num, which is equivalent to the number of clusters. Each of these 
        # entries are a list of tuples. The tuples are:
        # (original user index 0-1999, feature index 0-9999) 
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in self.m:
            if i[2] ==1:
                self.pos_index[self.groups[i[0]]].append((i[0], i[1]))
            else:
                self.neg_index[self.groups[i[0]]].append((i[0], i[1]))   

   
    def step(self):    
        # get a random user (in my case I can go in order of the users in the data)
        u = np.random.choice(range(2000))
        # Get the user group (which is the cluster of all similar users) to use as the
        # "real" user
        g = self.groups[u]
        # randomly placing the position of the reward
        arm = np.random.choice(range(10))

        # Not sure what these are yet exactly. Might just be setting up a scenario where 
        # one arm is 1 and the other 9 arns are 0. I might be able to skip past this part
        # if my data is already set up correctly for reward
        p_d = len(self.pos_index[g])
        n_d = len(self.neg_index[g])

        # These are (user, user feature) tuples
        pos = np.array(self.pos_index[g])[np.random.choice(range(p_d), 9, replace=True)]
        # This is 1 index that I assume will be a 1. It is randonly selected
        neg = np.array(self.neg_index[g])[np.random.choice(range(n_d), replace=True)]

        # This is just a concatenation of pos and neg
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0)

        X = []
        for ind in X_ind:
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(self.I[ind[1]])

        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        contexts = norm.pdf(np.array(X), loc=0, scale=0.5)
        # Return the user (integer),
        # the contexts (10 elements array, each element is a 10-element array. 
        # I think this gives one context),
        # and the reward (10 element array with 9 0s and 1 1)
        return g, contexts, rwd
    
class load_eedi_DALOCB:
    def __init__(self):
        def parse_data(filename):
            data = pd.read_csv(filename)
            return data

        def make_sparse_array(subjects_to_mark_one, length):
            arr = np.zeros(length)
            for subject in subjects_to_mark_one:
                arr[subject] = 1
            return list(arr)

        # TODO this might be the issue. Just calculate this as needed
        # and then overwrite it maybe?
        def create_context_all_questions(all_questions, question_subject_map, subject_subject_map):
            context_map = {}
            for question_id in all_questions:
                associated_subjects = question_subject_map[question_id]
                # print(question_id)
                # print(associated_subjects)
                subjects_to_mark_one = []
                for subject in associated_subjects:
                    subjects_to_mark_one.append(subject_subject_map[subject])
                context_row = make_sparse_array(subjects_to_mark_one, 388)
                context_map[question_id] = context_row
            return context_map

        def preprocess_dataset(df, question_subject_map, subject_subject_map):
            user_question_correct_tuple_list = []
            all_users = set()
            all_questions = set()

            for row in df.itertuples():
                question_id = row[1]
                user_id = row[2]
                is_correct = row[4]
                
                all_users.add(user_id)
                all_questions.add(question_id)
                user_question_correct_tuple_list.append( (user_id, question_id, is_correct) )

            context = create_context_all_questions(all_questions, question_subject_map, subject_subject_map)
            
            return user_question_correct_tuple_list, list(all_users), context

        def calculate_top_users(df, question_subject_map, num_users):
            student_dict = {}
            for row in df.itertuples():
                question_id = row[1]
                user_id = row[2]
                is_correct = row[4]
                                
                if user_id in student_dict:
                    student_dict[user_id].add_question_answered(question_id, is_correct, question_subject_map[question_id])
                else:
                    student_dict[user_id] = Student(user_id=user_id)
                    student_dict[user_id].add_question_answered(question_id, is_correct, question_subject_map[question_id])


            questions_answered_per_student = {}
            for s in student_dict.items():
                student = s[1]
                questions_answered_per_student[student.user_id] = len(student.questions_answered)
            questions_answered_per_student_df = pd.DataFrame(list(questions_answered_per_student.items()), columns = ['student_id','questions_answered'])
            new_one = questions_answered_per_student_df.sort_values('questions_answered', ascending = False)

            top_100 = new_one.head(num_users)
            print(top_100.shape)
            print(top_100)

            top_100_ids = top_100['student_id'].tolist()

            for id in top_100_ids:
                print(len(student_dict[id].questions_answered))

            return student_dict, top_100_ids

        # Question metadata
        question_metadata_df = parse_data("./public_data/metadata/question_metadata_task_1_2.csv")
        self.question_subject_map = {}
        for row in question_metadata_df.itertuples():
            question_id = row[1]
            subjects_str = row[2]
            subjects_list = ast.literal_eval(subjects_str)
            self.question_subject_map[question_id] = subjects_list

        # Subject metadata
        subject_metadata_df = parse_data("./public_data/metadata/subject_metadata.csv")
        # This is a mapping from subject_id to 0-387 index for simplicity and efficiency
        self.subjects = {}
        for row in subject_metadata_df.itertuples():
            subject_id = row[1]
            self.subjects[subject_id] = row[0]

        # Training data    
        self.all_data_df = parse_data("./public_data/train_data/train_task_1_2.csv")


        load_everything = True
        number_of_users_to_pull = 50
        if load_everything:
            # self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map)    

            with open('student_dict.pickle' , 'rb') as handle:
                self.student_dict = pickle.load(handle)   
            with open('top_ids' + str(number_of_users_to_pull) + '.pickle', 'rb') as handle:
                self.top_ids = pickle.load(handle) 

        else:
            self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map, number_of_users_to_pull)    
            with open('student_dict.pickle', 'wb') as handle:
                pickle.dump(self.student_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('top_ids' + str(number_of_users_to_pull) + '.pickle', 'wb') as handle:
                pickle.dump(self.top_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            exit()
        # Use top_ids to create new df to feed into main alg
        filtered_df = self.all_data_df[self.all_data_df.UserId.isin(self.top_ids)]

        # preprocess dataset. TODO could maybe save some of these as npy files
        user_question_correct_tuple_list, self.all_users, self.full_context = preprocess_dataset(filtered_df, self.question_subject_map, self.subjects)        

        self.num_user = len(self.all_users)
        self.dim = 388

        # Set up the lists of which users got which questions correctly
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in user_question_correct_tuple_list:
            if i[2] == 1:
                self.pos_index[i[0]].append(i[1])
            else:
                self.neg_index[i[0]].append(i[1])  



        # ---------
        # Get dictionary of students
        # self.student_dict = print_dataset_stats(self.all_data_df, self.question_subject_map)
        # Sample some students to test
        # sampled_students = random.sample(range(0, len(self.student_dict)), 5)
        # for s in sampled_students:  
        #     student = self.student_dict[s]
        #     print(student.print_info())
        # ---------

    def step(self):
        # This is 0-99
        user = np.random.choice(self.num_user)
        # Mapping to real user_id
        actual_user_id = self.top_ids[user]







        # ------
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[actual_user_id])
        n_d = len(self.neg_index[actual_user_id])


        # These are all question indices
        pos = np.array(self.pos_index[actual_user_id])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[actual_user_id])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1

        # I probably don't need to do this since my vars are not continous
        # contexts = norm.pdf(np.array(X), loc=0, scale=0.5)


        # ------
        # Here begins the dynamics calculations
        # ------


        # Need to add to a knowledge graph each time a question is considered
        question_to_add_index = np.random.choice(range(10))
        question_to_add_id = arms[question_to_add_index]
        user_correct = rwd[question_to_add_index]

        # Start here: Use new DA_LOCB and DA_Cluster classes and start in the __init__
        # for DA_LOCB. Create clusters randomly, perhaps (maybe not, keep reading). Use existing student class.
        # For actual clustering, inside student objects, add a field for the previous 
        # subject accuracies that update everytime a Q is answered. At every 1000 timesteps,
        # compare previous to current and if a subject accuracy is within delta (make this hyperparam)
        # Then it is static. Otherwise dynamic. Then do Kmeans to cluster all users based on 
        # the dynamics and change the clusters. Only issue is how exactly to update the math
        # in the clusters. Maybe I can do an inital cluster at t=1000 where everyone gets sorted with Kmeans. Then,
        # each 1000 timesteps, compute average change of each cluster (average all the users' current dynamics calc),
        # and check if each student should move to a new one or stay. Can then update math
        # Also only allow each student to be in 1 cluster\


        # The accuracies become more static over time because of the law of large numbers
        # Sp maybe I need to calculate accuracies in a sliding window. So every 20 qs 
        # the current,previous, and diff are overwritten and we start fresh




        return user, contexts, rwd, question_to_add_id, user_correct


class load_eedi_LOCB:
    def __init__(self):
        def parse_data(filename):
            data = pd.read_csv(filename)
            return data

        def make_sparse_array(subjects_to_mark_one, length):
            arr = np.zeros(length)
            for subject in subjects_to_mark_one:
                arr[subject] = 1
            return list(arr)

        # TODO this might be the issue. Just calculate this as needed
        # and then overwrite it maybe?
        def create_context_all_questions(all_questions, question_subject_map, subject_subject_map):
            context_map = {}
            for question_id in all_questions:
                associated_subjects = question_subject_map[question_id]
                # print(question_id)
                # print(associated_subjects)
                subjects_to_mark_one = []
                for subject in associated_subjects:
                    subjects_to_mark_one.append(subject_subject_map[subject])
                context_row = make_sparse_array(subjects_to_mark_one, 388)
                context_map[question_id] = context_row
            return context_map

        def preprocess_dataset(df, question_subject_map, subject_subject_map):
            user_question_correct_tuple_list = []
            all_users = set()
            all_questions = set()

            for row in df.itertuples():
                question_id = row[1]
                user_id = row[2]
                is_correct = row[4]
                
                all_users.add(user_id)
                all_questions.add(question_id)
                user_question_correct_tuple_list.append( (user_id, question_id, is_correct) )

            context = create_context_all_questions(all_questions, question_subject_map, subject_subject_map)
            
            return user_question_correct_tuple_list, list(all_users), context

        def calculate_top_users(df, question_subject_map, num_users):
            student_dict = {}
            for row in df.itertuples():
                question_id = row[1]
                user_id = row[2]
                is_correct = row[4]
                                
                if user_id in student_dict:
                    student_dict[user_id].add_question_answered(question_id, is_correct, question_subject_map[question_id])
                else:
                    student_dict[user_id] = Student(user_id=user_id)
                    student_dict[user_id].add_question_answered(question_id, is_correct, question_subject_map[question_id])


            questions_answered_per_student = {}
            for s in student_dict.items():
                student = s[1]
                questions_answered_per_student[student.user_id] = len(student.questions_answered)
            questions_answered_per_student_df = pd.DataFrame(list(questions_answered_per_student.items()), columns = ['student_id','questions_answered'])
            new_one = questions_answered_per_student_df.sort_values('questions_answered', ascending = False)

            top_100 = new_one.head(num_users)
            print(top_100.shape)
            print(top_100)

            top_100_ids = top_100['student_id'].tolist()

            for id in top_100_ids:
                print(len(student_dict[id].questions_answered))

            return student_dict, top_100_ids

        # Question metadata
        question_metadata_df = parse_data("./public_data/metadata/question_metadata_task_1_2.csv")
        self.question_subject_map = {}
        for row in question_metadata_df.itertuples():
            question_id = row[1]
            subjects_str = row[2]
            subjects_list = ast.literal_eval(subjects_str)
            self.question_subject_map[question_id] = subjects_list

        # Subject metadata
        subject_metadata_df = parse_data("./public_data/metadata/subject_metadata.csv")
        # This is a mapping from subject_id to 0-387 index for simplicity and efficiency
        self.subjects = {}
        for row in subject_metadata_df.itertuples():
            subject_id = row[1]
            self.subjects[subject_id] = row[0]

        # Training data    
        self.all_data_df = parse_data("./public_data/train_data/train_task_1_2.csv")


        load_everything = True
        number_of_users_to_pull = 50
        if load_everything:
            # self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map)    

            with open('student_dict.pickle' , 'rb') as handle:
                self.student_dict = pickle.load(handle)   
            with open('top_ids' + str(number_of_users_to_pull) + '.pickle', 'rb') as handle:
                self.top_ids = pickle.load(handle) 
            # print(student_dict == self.student_dict)
            # print(top_ids == self.top_ids)

            # for id in top_ids:
            #     print("--------")
            #     print(len(student_dict[id].questions_answered))
            #     print(len(self.student_dict[id].questions_answered))

        else:
            self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map, number_of_users_to_pull)    
            with open('student_dict.pickle', 'wb') as handle:
                pickle.dump(self.student_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('top_ids' + str(number_of_users_to_pull) + '.pickle', 'wb') as handle:
                pickle.dump(self.top_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            exit()
        # Use top_ids to create new df to feed into main alg
        filtered_df = self.all_data_df[self.all_data_df.UserId.isin(self.top_ids)]


        # preprocess dataset. TODO could maybe save some of these as npy files
        user_question_correct_tuple_list, self.all_users, self.full_context = preprocess_dataset(filtered_df, self.question_subject_map, self.subjects)        
        
        self.num_user = len(self.all_users)
        self.dim = 388

        # Set up the lists of which users got which questions correctly
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in user_question_correct_tuple_list:
            if i[2] == 1:
                self.pos_index[i[0]].append(i[1])
            else:
                self.neg_index[i[0]].append(i[1])  

    def step(self):
        # This is 0-99
        user = np.random.choice(self.num_user)
        # Mapping to real user_id
        actual_user_id = self.top_ids[user]

        # ------
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[actual_user_id])
        n_d = len(self.neg_index[actual_user_id])


        # These are all question indices
        pos = np.array(self.pos_index[actual_user_id])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[actual_user_id])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1

        # Need to add to a knowledge graph each time a question is considered
        # question_to_add_index = np.random.choice(range(10))
        # question_to_add_id = arms[question_to_add_index]
        # user_correct = rwd[question_to_add_index]

        return user, contexts, rwd



class load_eedi_prior_knowledge:
    def __init__(self):
        def parse_data(filename):
            data = pd.read_csv(filename)
            return data

        def make_sparse_array(subjects_to_mark_one, length):
            arr = np.zeros(length)
            for subject in subjects_to_mark_one:
                arr[subject] = 1
            return list(arr)

        # TODO this might be the issue. Just calculate this as needed
        # and then overwrite it maybe?
        def create_context_all_questions(all_questions, question_subject_map, subject_subject_map):
            context_map = {}
            for question_id in all_questions:
                associated_subjects = question_subject_map[question_id]
                # print(question_id)
                # print(associated_subjects)
                subjects_to_mark_one = []
                for subject in associated_subjects:
                    subjects_to_mark_one.append(subject_subject_map[subject])
                context_row = make_sparse_array(subjects_to_mark_one, 388)
                context_map[question_id] = context_row
            return context_map

        def preprocess_dataset(df, question_subject_map, subject_subject_map):
            user_question_correct_tuple_list = []
            all_users = set()
            all_questions = set()

            for row in df.itertuples():
                question_id = row[1]
                user_id = row[2]
                is_correct = row[4]
                
                all_users.add(user_id)
                all_questions.add(question_id)
                user_question_correct_tuple_list.append( (user_id, question_id, is_correct) )

            context = create_context_all_questions(all_questions, question_subject_map, subject_subject_map)
            
            return user_question_correct_tuple_list, list(all_users), context

        def calculate_top_users(df, question_subject_map, num_users):
            student_dict = {}
            for row in df.itertuples():
                question_id = row[1]
                user_id = row[2]
                is_correct = row[4]
                                
                if user_id in student_dict:
                    student_dict[user_id].add_question_answered(question_id, is_correct, question_subject_map[question_id])
                else:
                    student_dict[user_id] = Student(user_id=user_id)
                    student_dict[user_id].add_question_answered(question_id, is_correct, question_subject_map[question_id])


            questions_answered_per_student = {}
            for s in student_dict.items():
                student = s[1]
                questions_answered_per_student[student.user_id] = len(student.questions_answered)
            questions_answered_per_student_df = pd.DataFrame(list(questions_answered_per_student.items()), columns = ['student_id','questions_answered'])
            new_one = questions_answered_per_student_df.sort_values('questions_answered', ascending = False)

            top_100 = new_one.head(num_users)
            print(top_100.shape)
            print(top_100)

            top_100_ids = top_100['student_id'].tolist()

            for id in top_100_ids:
                print(len(student_dict[id].questions_answered))

            return student_dict, top_100_ids

        # Question metadata
        question_metadata_df = parse_data("./public_data/metadata/question_metadata_task_1_2.csv")
        self.question_subject_map = {}
        for row in question_metadata_df.itertuples():
            question_id = row[1]
            subjects_str = row[2]
            subjects_list = ast.literal_eval(subjects_str)
            self.question_subject_map[question_id] = subjects_list

        # Subject metadata
        subject_metadata_df = parse_data("./public_data/metadata/subject_metadata.csv")
        # This is a mapping from subject_id to 0-387 index for simplicity and efficiency
        self.subjects = {}
        for row in subject_metadata_df.itertuples():
            subject_id = row[1]
            self.subjects[subject_id] = row[0]

        # Training data    
        self.all_data_df = parse_data("./public_data/train_data/train_task_1_2.csv")


        load_everything = True
        number_of_users_to_pull = 50
        if load_everything:
            # self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map)    

            with open('student_dict.pickle' , 'rb') as handle:
                self.student_dict = pickle.load(handle)   
            with open('top_ids' + str(number_of_users_to_pull) + '.pickle', 'rb') as handle:
                self.top_ids = pickle.load(handle) 

        else:
            self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map, number_of_users_to_pull)    
            with open('student_dict.pickle', 'wb') as handle:
                pickle.dump(self.student_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('top_ids' + str(number_of_users_to_pull) + '.pickle', 'wb') as handle:
                pickle.dump(self.top_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
            exit()
        # Use top_ids to create new df to feed into main alg
        filtered_df = self.all_data_df[self.all_data_df.UserId.isin(self.top_ids)]

        # preprocess dataset. TODO could maybe save some of these as npy files
        user_question_correct_tuple_list, self.all_users, self.full_context = preprocess_dataset(filtered_df, self.question_subject_map, self.subjects)        

        self.num_user = len(self.all_users)
        self.dim = 388

        # Set up the lists of which users got which questions correctly
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in user_question_correct_tuple_list:
            if i[2] == 1:
                self.pos_index[i[0]].append(i[1])
            else:
                self.neg_index[i[0]].append(i[1])  

    def step(self):
        # This is 0-99
        user = np.random.choice(self.num_user)
        # Mapping to real user_id
        actual_user_id = self.top_ids[user]







        # ------
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[actual_user_id])
        n_d = len(self.neg_index[actual_user_id])


        # These are all question indices
        pos = np.array(self.pos_index[actual_user_id])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[actual_user_id])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1

        # I probably don't need to do this since my vars are not continous
        # contexts = norm.pdf(np.array(X), loc=0, scale=0.5)


        # ------
        # Here begins the dynamics calculations
        # ------


        # Need to add to a knowledge graph each time a question is considered
        question_to_add_index = np.random.choice(range(10))
        question_to_add_id = arms[question_to_add_index]
        user_correct = rwd[question_to_add_index]

        # Start here: Use new DA_LOCB and DA_Cluster classes and start in the __init__
        # for DA_LOCB. Create clusters randomly, perhaps (maybe not, keep reading). Use existing student class.
        # For actual clustering, inside student objects, add a field for the previous 
        # subject accuracies that update everytime a Q is answered. At every 1000 timesteps,
        # compare previous to current and if a subject accuracy is within delta (make this hyperparam)
        # Then it is static. Otherwise dynamic. Then do Kmeans to cluster all users based on 
        # the dynamics and change the clusters. Only issue is how exactly to update the math
        # in the clusters. Maybe I can do an inital cluster at t=1000 where everyone gets sorted with Kmeans. Then,
        # each 1000 timesteps, compute average change of each cluster (average all the users' current dynamics calc),
        # and check if each student should move to a new one or stay. Can then update math
        # Also only allow each student to be in 1 cluster\


        # The accuracies become more static over time because of the law of large numbers
        # Sp maybe I need to calculate accuracies in a sliding window. So every 20 qs 
        # the current,previous, and diff are overwritten and we start fresh




        return user, contexts, rwd, question_to_add_id, user_correct
