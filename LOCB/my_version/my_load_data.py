import numpy as np
from collections import defaultdict
import pandas as pd
from student_model import Student
import ast
import pickle
from os.path import exists
from os import listdir
import shutil


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

                shutil.copy2("../../KT1/" + filename, "./files_to_save/")
            if len(user_dfs) == 50:
                break
        exit()

        # Load question metadata
        question_metadata = parse_data("../../contents/questions.csv")

        user_question_correct_tuple_list, self.all_users = preprocess_dataset(user_dfs, user_ids, question_metadata)        
        self.full_context, self.question_subject_map = create_context(question_metadata)

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)

        for i in user_question_correct_tuple_list:
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

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)

        for i in user_question_correct_tuple_list:
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

        pos = np.array(self.pos_index[user])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[user])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1
        
        return user, contexts, rwd
       
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

            top_users = new_one.head(num_users)
            top_user_ids = top_users['student_id'].tolist()

            return student_dict, top_user_ids

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

        filtered_df = self.all_data_df[self.all_data_df.UserId.isin(self.top_ids)]

        user_question_correct_tuple_list, self.all_users, self.full_context = preprocess_dataset(filtered_df, self.question_subject_map, self.subjects)        

        self.num_user = len(self.all_users)
        self.dim = 388

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in user_question_correct_tuple_list:
            if i[2] == 1:
                self.pos_index[i[0]].append(i[1])
            else:
                self.neg_index[i[0]].append(i[1])  

    def step(self):
        user = np.random.choice(self.num_user)
        actual_user_id = self.top_ids[user]
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[actual_user_id])
        n_d = len(self.neg_index[actual_user_id])

        pos = np.array(self.pos_index[actual_user_id])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[actual_user_id])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1

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
            
        filtered_df = self.all_data_df[self.all_data_df.UserId.isin(self.top_ids)]

        user_question_correct_tuple_list, self.all_users, self.full_context = preprocess_dataset(filtered_df, self.question_subject_map, self.subjects)        
        
        self.num_user = len(self.all_users)
        self.dim = 388

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in user_question_correct_tuple_list:
            if i[2] == 1:
                self.pos_index[i[0]].append(i[1])
            else:
                self.neg_index[i[0]].append(i[1])  

    def step(self):
        user = np.random.choice(self.num_user)
        actual_user_id = self.top_ids[user]
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[actual_user_id])
        n_d = len(self.neg_index[actual_user_id])

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

            top_users = new_one.head(num_users)
            print(top_users.shape)
            print(top_users)

            top_user_ids = top_users['student_id'].tolist()

            for id in top_user_ids:
                print(len(student_dict[id].questions_answered))

            return student_dict, top_user_ids

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

        filtered_df = self.all_data_df[self.all_data_df.UserId.isin(self.top_ids)]

        user_question_correct_tuple_list, self.all_users, self.full_context = preprocess_dataset(filtered_df, self.question_subject_map, self.subjects)        

        self.num_user = len(self.all_users)
        self.dim = 388

        self.pos_index = defaultdict(list)
        self.neg_index = defaultdict(list)
        for i in user_question_correct_tuple_list:
            if i[2] == 1:
                self.pos_index[i[0]].append(i[1])
            else:
                self.neg_index[i[0]].append(i[1])  

    def step(self):
        user = np.random.choice(self.num_user)
        actual_user_id = self.top_ids[user]
        arm = np.random.choice(range(10))

        p_d = len(self.pos_index[actual_user_id])
        n_d = len(self.neg_index[actual_user_id])

        pos = np.array(self.pos_index[actual_user_id])[np.random.choice(range(p_d), replace=True)]
        neg = np.array(self.neg_index[actual_user_id])[np.random.choice(range(n_d), 9, replace=True)]
        arms = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0)

        contexts = []
        for question_id in arms:
            contexts.append(np.array(self.full_context[question_id]))
        
        rwd = np.zeros(10)
        rwd[arm] = 1

        question_to_add_index = np.random.choice(range(10))
        question_to_add_id = arms[question_to_add_index]
        user_correct = rwd[question_to_add_index]

        return user, contexts, rwd, question_to_add_id, user_correct
