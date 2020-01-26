
import numpy as np

class User:
    def __init__(self, user_id, model, averaging_method, averaging_metric,
                 train_class, train_data,
                 val_class, val_data,
                 test_class, test_data):
        self._id = user_id
        self._model = model
        self._history = []
        self._history_metrics = None
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

        self._averaging_metric = averaging_metric
        self._averaging_method = averaging_method #std_dev, weighted_avg

    def get_averaging_metric(self):
        return self._averaging_metric

    def set_averaging_metric(self, averaging_metric):
        self._averaging_metric = averaging_metric

    def set_averaging_method(self, averaging_method):
        self._averaging_method = averaging_method

    def get_averaging_method(self):
        return self._averaging_method

    def evaluate(self, verbose = True):
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
        trains the model for the user instance
        and updates the weights and history attribute for the user too
        """

        train_data = self.get_train_data()
        train_class = self.get_train_class()
        val_data = self.get_val_data()
        val_class = self.get_val_class()
        model = self.get_model()
        if weights == None: # if provided, update model weights
            pass
        elif type(weights) == type(dict()): # personalised strat
            original_user_weights = self.get_weights()
            # you save the original weights and then calcualte new weights
            # based on the strat set for the user
            new_weights = self.get_averaging_method()(user = self,
                                        weights = weights,
                                        metric = self.get_averaging_metric())
            model.set_weights(new_weights)
            print("New weights",np.array_equal(original_user_weights, new_weights))
        else:
            model.set_weights(weights)

        e = self.evaluate(verbose = verbose_evaluate)
        self.add_pre_fit_evaluation(e)
        print(self.get_id(), e)
        # sanity check to see they all have the same init weights
        # print(self.get_id())
        # print(model.get_weights())


        history = model.fit(
            train_data,
            train_class,
            epochs = epochs,
            verbose = verbose_fit,
            shuffle = False,
            # batch_size = 2**8, #4k
            # use_multiprocessing = True,
            validation_data = (val_data, val_class)
        )

        e = self.evaluate(verbose = verbose_evaluate)
        self.add_post_fit_evaluation(e)

        # update user data
        self.add_history(history)

        return


    def get_data(self, ignore_first_n = 0,
                 metric = "accuracy",
                 pre = False, post = False):
        if (metric not in ["accuracy", "loss"]) or (pre == post):

            raise Exception("Please select one from \
                    accuracy or loss and one from pre or post")
        if metric == "loss":
            if pre:
                data = self.get_pre_fit_loss()
            else:
                data = self.get_post_fit_loss()
        elif metric == "accuracy":
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
        return self._train_class

    def get_train_data(self):
        return self._train_data

    def get_val_class(self):
        return self._val_class

    def get_val_data(self):
        return self._val_data

    def get_test_class(self):
        return self._test_class

    def get_test_data(self):
        return self._test_data

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

    def add_train_class(self, train_class):
        self._train_class = np.concatenate((self._train_class, train_class))

    def add_train_data(self, train_data):
        self._train_data = np.concatenate((self._train_data, train_data))

    def add_val_class(self, val_class):
        self._val_class = np.concatenate((self._val_class, val_class))

    def add_val_data(self, val_data):
        self._val_data = np.concatenate((self._val_data, val_data))

    def add_test_class(self, test_class):
        self._test_class = np.concatenate((self._test_class, test_class))

    def add_test_data(self, test_data):
        self._test_data = np.concatenate((self._test_data, test_data))

    def get_history(self):
        return self._history

    def add_history(self, history):
        self._history.append(history)
        self.add_history_metrics(history)
        
    def add_history_metrics(self, history):
        if self._history_metrics == None:
            self._history_metrics = history.history
        else:
            for key in self._history_metrics.keys():
                self._history_metrics[key] = self._history_metrics[key] + history.history[key]
        
    def get_history_metrics(self):
        return self._history_metrics
        
    def get_weights(self):
        return self.get_model().get_weights()

    def set_weights(self, weights):
        self.get_model().set_weights(weights)
