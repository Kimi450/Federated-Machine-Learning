
import numpy as np

class User:
    def __init__(self, user_id, averaging_method, model,
    train_class, train_data,
    val_class, val_data,
    test_class, test_data):
        self._id = user_id
        self._model = model
        self._history = []
        self._train_class = train_class
        self._train_data = train_data
        self._val_class = val_class
        self._val_data = val_data
        self._test_class = test_class
        self._test_data = test_data

        self._post_fit_loss = np.array([])
        self._post_fit_accuracy = np.array([])
        self._pre_fit_loss = np.array([])
        self._pre_fit_accuracy = np.array([])
    
        self._averaging_method = averaging_method #std_dev, weighted_avg
        
    def set_averaging_method(self, averaging_method):
        self._averaging_method = averaging_method
        
    def get_averaging_method(self):
        return self._averaging_method
        

    
        
        
    def evaluate_user(self, verbose = True):
        """returns the loss and accuracy for the given User instance
        on test data"""
        test_data = self.get_test_data()
        test_class = self.get_test_class()
        model = self.get_model()
        evaluation = model.evaluate(test_data,
                                    test_class,
                                    verbose = verbose)
        return evaluation

    def train(self, epochs=16, weights=None, verbose_fit = False,
              verbose_evaluate = False):
        # https://www.tensorflow.org/beta/tutorials/keras/basic_classification
        # same seed value for consistency sake, across all trainings too
        """
        trains the model for the user
        and updates the weights and history attribute for the user too
        """

        train_data = self.get_train_data()
        train_class = self.get_train_class()
        val_data = self.get_val_data()
        val_class = self.get_val_class()
        model = self.get_model()

        if weights != None: # if provided, update model weights
            model.set_weights(weights)

        e = self.evaluate_user(verbose = verbose_evaluate)
        self.add_pre_fit_evaluation(e)


        history = model.fit(
            train_data,
            train_class,
            epochs = epochs,
            verbose = verbose_fit,
            # batch_size = 2**8, #4k
            # use_multiprocessing = True,
            validation_data = (val_data, val_class)
        )

        e = self.evaluate_user(verbose = verbose_evaluate)
        self.add_post_fit_evaluation(e)

        # update user data
        self.add_history(history)

        return

        
    def get_data(self, ignore_first_n = 0, 
                 loss = False, accuracy = False, 
                 pre = False, post = False):
        if (loss == accuracy) or (pre == post):

            print("Please select one from accuracy or loss and one from pre or post")
            return None
        if loss:
            if pre:
                data = self.get_pre_fit_loss()
            else:
                data = self.get_post_fit_loss()
        else:
            if pre:
                data = self.get_pre_fit_accuracy()
            else:
                data = self.get_post_fit_accuracy()
        if ignore_first_n > 0:
            data = data[ignore_first_n:]
        return data

    def get_pre_fit_loss(self):
        return self._pre_fit_loss
    def get_pre_fit_accuracy(self):
        return self._pre_fit_accuracy
    def get_post_fit_accuracy(self):
        return self._post_fit_accuracy
    def get_post_fit_loss(self):
        return self._post_fit_loss

    def get_latest_accuracy(self, pre, post):
        data = None
        if pre:
            data = self.get_pre_fit_accuracy()
        elif post:
            data = self.get_post_fit_accuracy()
        else:
            print("Please select one of pre or post as True")
            return None
        return data[-1]
    
    def get_latest_loss(self, pre, post):
        data = None
        
        if pre:
            data = self.get_pre_fit_loss()
        elif post:
            data = self.get_post_fit_loss()
        else:
            print("Please select one of pre or post as True")
            return None
        return data[-1]
    
    def add_pre_fit_evaluation(self, pre_fit_evaluation):
        self._pre_fit_loss = np.append(self.get_pre_fit_loss(),
                                       pre_fit_evaluation[0])
        self._pre_fit_accuracy = np.append(self.get_pre_fit_accuracy(),
                                           pre_fit_evaluation[1])

    def add_post_fit_evaluation(self,post_fit_evaluation):
        self._post_fit_loss = np.append(self.get_post_fit_loss(),
                                        post_fit_evaluation[0])
        self._post_fit_accuracy = np.append(self.get_post_fit_accuracy(),
                                            post_fit_evaluation[1])

    def get_model(self):
        return self._model

    def get_id(self):
        return self._id

    def get_train_class(self):
        return self._train_class.values

    def get_train_data(self):
        return self._train_data.values

    def get_val_class(self):
        return self._val_class.values

    def get_val_data(self):
        return self._val_data.values

    def get_test_class(self):
        return self._test_class.values

    def get_test_data(self):
        return self._test_data.values

    def set_model(self, model):
        self._model = model

    def set_id(self, id):
        self._id = id

    def set_train_class(self, train_class):
        self._train_class = train_class

    def set_train_data(self, train_data):
        self._train_data = train_data

    def set_val_class(self, val_class):
        self._val_class = val_class

    def set_val_data(self, val_data):
        self._val_data = val_data

    def set_test_class(self, test_class):
        self._test_class = test_class

    def set_test_data(self, test_data):
        self._test_data = test_data




    def get_history(self):
        return self._history

    def add_history(self, history):
        self._history.append(history)

    def get_weights(self):
        return self.get_model().get_weights()

    def set_weights(self, weights):
        self.get_model().set_weights(weights)
