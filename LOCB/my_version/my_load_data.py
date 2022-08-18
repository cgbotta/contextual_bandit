from dataclasses import replace
from email.utils import parsedate
from re import sub
from typing import AsyncGenerator
import operator
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
import math
from scipy.stats import norm
import pickle
from os.path import exists, isfile, join
from os import listdir

class load_ednet_DALOCB:
    def __init__(self):
        def parse_data(filename):
            data = pd.read_csv(filename)
            return data

        def make_sparse_array(subjects_to_mark_one, length):
            arr = np.zeros(length)
            for subject in subjects_to_mark_one:
                arr[subject] = 1
            return list(arr)
        
        def preprocess_dataset(user_dfs, user_ids, question_metadata):
            user_question_correct_tuple_list = []
            all_users = set()
            # all_questions = set()

            for index, user_df in enumerate(user_dfs):
                user_id = user_ids[index]
                all_users.add(user_id)

                for row in user_df.itertuples():
                    question_id = row[3]
                    user_answer = row[4]
                    correct_answer_row = question_metadata.loc[question_metadata['question_id'] == question_id]
                    correct_answer = correct_answer_row.iloc[0]['correct_answer']

                    if user_answer == correct_answer:
                        isCorrect = 1
                    else:
                        isCorrect = 0

                    # all_questions.add(question_id)
                    user_question_correct_tuple_list.append( (index, question_id, isCorrect) )
            
            all_users = list(all_users)
            return user_question_correct_tuple_list, all_users

        def create_context(question_metadata):
            context_map = {}
            question_subject_map = {}

            for row in question_metadata.itertuples():
                question_id = row[1]
                associated_subjects = row[6].split(';')
                subjects_to_mark_one = []
                for subject in associated_subjects:
                    subject = int(subject)
                    if subject == -1:
                        continue
                    subjects_to_mark_one.append(subject)
                question_subject_map[question_id] = subjects_to_mark_one
                context_row = make_sparse_array(subjects_to_mark_one, 301)
                context_map[question_id] = context_row
            return context_map, question_subject_map            

        file_list = listdir("../../KT1")
        user_dfs = []
        user_ids = []
        for filename in file_list:
            df = parse_data("../../KT1/" + filename)
            if df.shape[0] >= 1000:
                user_dfs.append(parse_data("../../KT1/" + filename))
                user_ids.append(filename.split('.')[0])
            if len(user_dfs) == 50:
                break

        # Load question metadata
        question_metadata = parse_data("../../contents/questions.csv")

        user_question_correct_tuple_list, self.all_users = preprocess_dataset(user_dfs, user_ids, question_metadata)        
        self.full_context, self.question_subject_map = create_context(question_metadata)

        # Set up the lists of which users answered which questions correctly
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)

        for i in user_question_correct_tuple_list:
            # Get user index
            index = i[0]
            if i[2] == 1:
                self.pos_index[index].append(i[1])
            else:
                self.neg_index[index].append(i[1]) 

        self.num_user = len(self.all_users)
        self.dim = 301
        self.subjects = {}
        for i in range(self.dim):
            self.subjects[i] = i

    def step(self):
        user = np.random.choice(range(len(self.all_users)))
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[user])
        n_d = len(self.neg_index[user])

        # These are all question indices
        pos = np.array(self.pos_index[user])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[user])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1

        # Need to add to a knowledge graph each time a question is considered
        question_to_add_index = np.random.choice(range(10))
        question_to_add_id = arms[question_to_add_index]
        user_correct = rwd[question_to_add_index]

        return user, contexts, rwd, question_to_add_id, user_correct

class load_ednet_LOCB:
    def __init__(self):
        def parse_data(filename):
            data = pd.read_csv(filename)
            return data

        def make_sparse_array(subjects_to_mark_one, length):
            arr = np.zeros(length)
            for subject in subjects_to_mark_one:
                arr[subject] = 1
            return list(arr)
        
        def preprocess_dataset(user_dfs, user_ids, question_metadata):
            user_question_correct_tuple_list = []
            all_users = set()
            # all_questions = set()

            for index, user_df in enumerate(user_dfs):
                user_id = user_ids[index]
                all_users.add(user_id)

                for row in user_df.itertuples():
                    question_id = row[3]
                    user_answer = row[4]
                    correct_answer_row = question_metadata.loc[question_metadata['question_id'] == question_id]
                    correct_answer = correct_answer_row.iloc[0]['correct_answer']

                    if user_answer == correct_answer:
                        isCorrect = 1
                    else:
                        isCorrect = 0

                    # all_questions.add(question_id)
                    user_question_correct_tuple_list.append( (index, question_id, isCorrect) )
            
            all_users = list(all_users)
            return user_question_correct_tuple_list, all_users

        def create_context(question_metadata):
            context_map = {}
            question_subject_map = {}

            for row in question_metadata.itertuples():
                question_id = row[1]
                associated_subjects = row[6].split(';')
                subjects_to_mark_one = []
                for subject in associated_subjects:
                    subject = int(subject)
                    if subject == -1:
                        continue
                    subjects_to_mark_one.append(subject)
                question_subject_map[question_id] = subjects_to_mark_one
                context_row = make_sparse_array(subjects_to_mark_one, 301)
                context_map[question_id] = context_row
            return context_map, question_subject_map            

        file_list = listdir("../../KT1")
        user_dfs = []
        user_ids = []
        for filename in file_list:
            df = parse_data("../../KT1/" + filename)
            if df.shape[0] >= 1000:
                user_dfs.append(parse_data("../../KT1/" + filename))
                user_ids.append(filename.split('.')[0])
            if len(user_dfs) == 50:
                break

        # Load question metadata
        question_metadata = parse_data("../../contents/questions.csv")

        user_question_correct_tuple_list, self.all_users = preprocess_dataset(user_dfs, user_ids, question_metadata)        
        self.full_context, self.question_subject_map = create_context(question_metadata)

        # Set up the lists of which users answered which questions correctly
        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)

        for i in user_question_correct_tuple_list:
            # Get user index
            index = i[0]
            if i[2] == 1:
                self.pos_index[index].append(i[1])
            else:
                self.neg_index[index].append(i[1]) 

        self.num_user = len(self.all_users)
        self.dim = 301
        self.subjects = {}
        for i in range(self.dim):
            self.subjects[i] = i

    def step(self):
        user = np.random.choice(range(len(self.all_users)))
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[user])
        n_d = len(self.neg_index[user])

        # These are all question indices
        pos = np.array(self.pos_index[user])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[user])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1
        
        return user, contexts, rwd

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

        def create_context_all_questions(all_questions, question_subject_map, subject_subject_map):
            context_map = {}
            for question_id in all_questions:
                associated_subjects = question_subject_map[question_id]
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
            top_100_ids = top_100['student_id'].tolist()

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

        #  User metadata
        user_metadata_df = parse_data("./public_data/metadata/student_metadata_task_1_2.csv")
        self.user_metadata_map = {}
        for row in user_metadata_df.itertuples():
            user_id = row[1]
            gender = row[2]
            data_row =  row[3]
            if not pd.isnull(data_row):
                age = 2020 - int(data_row.split('-')[0])
            else:
                age = 0
            premium_pupil_row = row[4]
            if not pd.isnull(premium_pupil_row):
                premium_pupil = premium_pupil_row
            else:
                premium_pupil = -1

            self.user_metadata_map[user_id] = (gender, age, premium_pupil)

        number_of_users_to_pull = 50
        if exists('student_dict_1_2.pickle') and exists('top_ids' + str(number_of_users_to_pull)+ '_1_2' + '.pickle'):
            with open('student_dict_1_2.pickle' , 'rb') as handle:
                self.student_dict = pickle.load(handle)   
            with open('top_ids' + str(number_of_users_to_pull)+ '_1_2' + '.pickle', 'rb') as handle:
                self.top_ids = pickle.load(handle) 
        else:
            self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map, number_of_users_to_pull)    
            with open('student_dict_1_2.pickle', 'wb') as handle:
                pickle.dump(self.student_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('top_ids' + str(number_of_users_to_pull) + '_1_2' + '.pickle', 'wb') as handle:
                pickle.dump(self.top_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("------------------------")
            print("Program execution stopped to save some of the data into .npy files in the current directory")
            print("Please rerun the program, the analysis will run much more qickly now!")
            print("------------------------")
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
        question_to_add_index = np.random.choice(range(10))
        question_to_add_id = arms[question_to_add_index]
        user_correct = rwd[question_to_add_index]

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

        def create_context_all_questions(all_questions, question_subject_map, subject_subject_map):
            context_map = {}
            for question_id in all_questions:
                associated_subjects = question_subject_map[question_id]
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
            top_100_ids = top_100['student_id'].tolist()

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

        number_of_users_to_pull = 50
        if exists('student_dict_1_2.pickle') and exists('top_ids' + str(number_of_users_to_pull)+ '_1_2' + '.pickle'):
            with open('student_dict_1_2.pickle' , 'rb') as handle:
                self.student_dict = pickle.load(handle)   
            with open('top_ids' + str(number_of_users_to_pull)+ '_1_2' + '.pickle', 'rb') as handle:
                self.top_ids = pickle.load(handle) 
        else:
            self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map, number_of_users_to_pull)    
            with open('student_dict_1_2.pickle', 'wb') as handle:
                pickle.dump(self.student_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open('top_ids' + str(number_of_users_to_pull) + '_1_2' + '.pickle', 'wb') as handle:
                pickle.dump(self.top_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("------------------------")
            print("Program execution stopped to save some of the data into .npy files in the current directory")
            print("Please rerun the program, the analysis will run much more qickly now!")
            print("------------------------")
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

        return user, contexts, rwd

class load_eedi_DALOCB_static:
    def __init__(self):
        def parse_data(filename):
            data = pd.read_csv(filename)
            return data

        def make_sparse_array(subjects_to_mark_one, length):
            arr = np.zeros(length)
            for subject in subjects_to_mark_one:
                arr[subject] = 1
            return list(arr)

        def create_context_all_questions(all_questions, question_subject_map, subject_subject_map):
            context_map = {}
            for question_id in all_questions:
                associated_subjects = question_subject_map[question_id]
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

        #  User metadata
        user_metadata_df = parse_data("./public_data/metadata/student_metadata_task_1_2.csv")
        self.user_metadata_map = {}
        for row in user_metadata_df.itertuples():
            user_id = row[1]
            gender = row[2]
            data_row =  row[3]
            if not pd.isnull(data_row):
                age = 2020 - int(data_row.split('-')[0])
            else:
                age = 0
            premium_pupil_row = row[4]
            if not pd.isnull(premium_pupil_row):
                premium_pupil = premium_pupil_row
            else:
                premium_pupil = -1

            self.user_metadata_map[user_id] = (gender, age, premium_pupil)

        number_of_users_to_pull = 50
        if exists('student_dict_1_2.pickle') and exists('top_ids' + str(number_of_users_to_pull)+ '_1_2' + '.pickle'):
            with open('student_dict_1_2.pickle' , 'rb') as handle:
                self.student_dict = pickle.load(handle)   
            with open('top_ids' + str(number_of_users_to_pull)+ '_1_2' + '.pickle', 'rb') as handle:
                self.top_ids = pickle.load(handle) 
        else:
            self.student_dict, self.top_ids = calculate_top_users(self.all_data_df, self.question_subject_map, number_of_users_to_pull)    
            with open('student_dict_1_2.pickle', 'wb') as handle:
                pickle.dump(self.student_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('top_ids' + str(number_of_users_to_pull) + '_1_2' + '.pickle', 'wb') as handle:
                pickle.dump(self.top_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("------------------------")
            print("Program execution stopped to save some of the data into .npy files in the current directory")
            print("Please rerun the program, the analysis will run much more qickly now!")
            print("------------------------")
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
        question_to_add_index = np.random.choice(range(10))
        question_to_add_id = arms[question_to_add_index]
        user_correct = rwd[question_to_add_index]

        return user, contexts, rwd, question_to_add_id, user_correct
