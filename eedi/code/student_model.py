class Student:
    def __init__(self, user_id):
        self.user_id = user_id
        self.subject_accuracy = [None] * 1989
        self.total_questions_answered = 0
        self.correct_answers = 0
        # Key is question ID, value is correct (1) or incorrect (0)
        self.questions_answered = {}

    def add_question_answered(self, question_id, correct, subjects):
        if question_id in self.questions_answered:
            print("Error: student " + str(self.user_id) + " answered question " + str(question_id) + " multiple times!")
        else:
            self.questions_answered[question_id] = correct
            self.total_questions_answered = self.total_questions_answered + 1
            if correct:
                self.correct_answers = self.correct_answers + 1
            for subject in subjects:
                if self.subject_accuracy[subject] is None:
                    performance = [0,0]
                    if correct:
                        performance = [1,1]
                    else:
                        performance = [0,1]
                    self.subject_accuracy[subject] = performance
                else:
                    previous_performance = self.subject_accuracy[subject]
                    if correct:
                        previous_performance[0] = previous_performance[0] + 1
                        previous_performance[1] = previous_performance[1] + 1
                    else:
                        previous_performance[1] = previous_performance[1] + 1


    def get_subject_features_binary(self):
        return

    def print_info(self):
        print("Correct answers:", self.correct_answers)
        print("Questions answered in total:",self.total_questions_answered)
        print("Subject Accuracy:", self.subject_accuracy)




