import numpy as np

class Student:
    def __init__(self, user_id):
        self.user_id = user_id
        self.subject_accuracy = {}
        self.total_questions_answered = 0
        self.correct_answers = 0
        self.questions_with_correctness = []
        self.questions_answered = {}
        
        self.current_knowledge_state = []
        self.previous_knowledge_state = []
        self.current_diff = []
        self.average_diff = 0
        self.total_diff = 0
        self.questions_since_last_diff = 0
        self.diffs_taken = 0

    def add_question_answered(self, question_id, correct, subjects):
        if question_id in self.questions_answered:
            return
        else:
            self.questions_answered[question_id] = correct
            self.total_questions_answered = self.total_questions_answered + 1
            if correct:
                self.correct_answers = self.correct_answers + 1
            for subject in subjects:
                previous_performance = self.subject_accuracy.get(subject)
                if previous_performance is None:
                    performance = [0,0]
                    if correct:
                        performance = [1,1]
                    else:
                        performance = [0,1]
                    self.subject_accuracy[subject] = performance
                else:
                    if correct:
                        previous_performance[0] = previous_performance[0] + 1
                        previous_performance[1] = previous_performance[1] + 1
                    else:
                        previous_performance[1] = previous_performance[1] + 1
            self.questions_with_correctness.append(( question_id, correct ))
    
    def add_question_answered_percentage(self, question_id, correct, subjects):
        if question_id in self.questions_answered:
            return
        else:
            self.questions_answered[question_id] = correct
            self.total_questions_answered = self.total_questions_answered + 1
            if correct:
                self.correct_answers = self.correct_answers + 1
            for subject in subjects:
                previous_performance = self.subject_accuracy.get(subject)
                if previous_performance is None:
                    performance = [0.0,0.0,0.0]
                    if correct:
                        performance = [1.0,1.0,1.0]
                    else:
                        performance = [0.0,1.0,0.0]
                    self.subject_accuracy[subject] = performance
                else:
                    if correct:
                        previous_performance[0] = previous_performance[0] + 1
                        previous_performance[1] = previous_performance[1] + 1
                        previous_performance[2] = round(previous_performance[0] / previous_performance[1],3)
                    else:
                        previous_performance[1] = previous_performance[1] + 1
                        previous_performance[2] = round(previous_performance[0] / previous_performance[1],3)

            self.questions_with_correctness.append(( question_id, correct ))

    def get_subject_accuracies_array(self, subject_map, user_update_frequency):
        self.questions_since_last_diff = self.questions_since_last_diff + 1

        if self.questions_since_last_diff > user_update_frequency:
            new = np.zeros(len(subject_map))
            for subject_id, value in sorted(self.subject_accuracy.items()):
                    new[subject_map[subject_id]] = value[2]
            self.previous_knowledge_state = self.current_knowledge_state
            self.current_knowledge_state = new

            if len(self.previous_knowledge_state) != 0:
                self.current_diff = np.absolute(np.subtract(self.current_knowledge_state, self.previous_knowledge_state))
            else:
                self.current_diff = self.current_knowledge_state
            self.questions_since_last_diff = 0
            self.diffs_taken = self.diffs_taken + 1
            diff_sum = sum(self.current_diff)
            self.total_diff = self.total_diff + diff_sum
            self.average_diff = self.total_diff / self.diffs_taken

            # Clearing this resets everything, so it is as if we are doing a sliding window
            self.subject_accuracy = {}

    def get_subject_accuracies_array_static(self, subject_map, user_metadata, user_update_frequency):
        self.questions_since_last_diff = self.questions_since_last_diff + 1

        if self.questions_since_last_diff > user_update_frequency:
            new = np.zeros(len(subject_map))
            for subject_id, value in sorted(self.subject_accuracy.items()):
                new[subject_map[subject_id]] = value[2]

            # Add static features
            metadata = user_metadata[self.user_id]
            for val in metadata:
                new = np.append(new, val)
                
            self.previous_knowledge_state = self.current_knowledge_state
            self.current_knowledge_state = new

            if len(self.previous_knowledge_state) != 0:
                self.current_diff = np.absolute(np.subtract(self.current_knowledge_state, self.previous_knowledge_state))
            else:
                self.current_diff = self.current_knowledge_state
            self.questions_since_last_diff = 0
            self.diffs_taken = self.diffs_taken + 1
            diff_sum = sum(self.current_diff)
            self.total_diff = self.total_diff + diff_sum
            self.average_diff = self.total_diff / self.diffs_taken

            # Clearing this resets everything, so it is as if we are doing a sliding window
            self.subject_accuracy = {}


    def print_info(self):
        print("Correct answers:", self.correct_answers)
        print("Questions answered in total:",self.total_questions_answered)
        print("Subject Accuracy:", self.subject_accuracy)
