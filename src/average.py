import numpy as np

class Average:
    def __init__(self):
        """
        this is used to average all the users data using 'method' method
        It returns the new weights as a list of numpy arrrays.
        all, std_dev or weighted_average.

        all is where all the users' weights are just averaged, google policy

        std_dev is where users with metrics 1 std_dev less than the average
        are discarded

        weighted_avg is the (weights*metric)/(sum of metrics from all users)
        """

    def _raise(ex):
        raise ex

    def _latest_user_metric(user, pre, post, accuracy, loss):
        data = None
        if accuracy:
            data = user.get_latest_accuracy(pre = pre, post = post)
        elif loss:
            data = user.get_latest_loss(pre = pre, post = post)
        else:
            raise Exception("Please select loss or accuracy as metric")
        return data

    def _initialise(user, loss, accuracy, post, pre):

        if type(user) == type(dict()):
            user = list(user.values())[0]

        if pre == post or loss==accuracy:
            raise Exception("Please select one of pre or post and loss or accuracy")

        users_used = set()
        #create a numpy array of 0s
        new_weights = np.asarray(user.get_weights())
        for i in new_weights:
            i[i==i] = 0

        return new_weights, users_used

    def all(users = None,
            user = None, weights = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights = None
        if users:
            new_weights = Average.all_central(users = users,
                loss = loss,
                accuracy = accuracy,
                post = post,
                pre = pre)

        elif user and weights:
            new_weights = Average.all_personalised(user = user,
                weights = weights,
                loss = loss,
                accuracy = accuracy,
                post = True,
                pre = False)
        return new_weights

    def all_central(users = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used = Average._initialise(users,
                                            loss = loss,
                                            accuracy = accuracy,
                                            post = post,
                                            pre = pre)
        for user in users.values():
            user_weights = np.asarray(user.get_weights())
            users_used.add(user.get_id())
            #nested array of [weights] and [biases]
            new_weights += user_weights
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()


    def all_personalised(user = None, weights = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used = Average._initialise(user,

                                            loss = loss,
                                            accuracy = accuracy,
                                            post = post,
                                            pre = pre)

        original_weights = user.get_weights()
        test_data = user.get_test_data()
        test_class = user.get_test_class()
        for user_id, weight in weights.items():
            user_weights = np.asarray(weight)
            users_used.add(user_id)
            #nested array of [weights] and [biases]
            new_weights += user_weights
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()

    def eval_user_on_all(user, weights, loss, accuracy):
        """
        returns a dictionary of user_id:metric where metric is
        accuracy or loss based on what boolean parameters were
        passed in, both are not allowed.
        it places weights from 'weights' in users current model and
        evaluates it on those custom weights models (recieved from all users)
        and tests on the users own test data, returning the metric values as
        a dictionary
        """
        if loss == accuracy:
            raise Exception("Please select one of loss or accuracy")
        evals = dict.fromkeys(list(weights.keys()))
        original_weights = user.get_weights()
        for user_id,weight in weights.items():
            user.set_weights(weight)
            metric = user.evaluate(verbose = False)
            if loss:
                metric = metric[0]
            elif accuracy:
                metric = metric[1]

            evals[user_id] = metric

        user.set_weights(original_weights)
        return evals

    def std_dev(users = None,
            user = None, weights = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights = None
        if users:
            new_weights = Average.std_dev_central(users = users,
                loss = loss,
                accuracy = accuracy,
                post = post,
                pre = pre)

        elif user and weights:
            new_weights = Average.std_dev_personalised(user = user,
                weights = weights,
                loss = loss,
                accuracy = accuracy,
                post = True,
                pre = False)
        return new_weights

    def std_dev_personalised(user = None, weights = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used = Average._initialise(user,
                                            loss = loss,
                                            accuracy = accuracy,
                                            post = post,
                                            pre = pre)

        evals = Average.eval_user_on_all(user = user, weights = weights,
                                 loss = loss, accuracy = accuracy)
        latest_metrics = list(evals.values())
        latest_metrics = np.asarray(latest_metrics)
        std_dev = latest_metrics.std()
        avg = latest_metrics.mean()

## from here
        for user_id, curr_metric in evals.items():
            if curr_metric >= (avg-std_dev):
                user_weights = np.asarray(weights[user_id])
                users_used.add(user_id)
                #nested array of [weights] and [biases]
                new_weights += user_weights
            else:
                output=f"User: {user.get_id()} -> User {user_id}: {curr_metric} < {avg-std_dev}"
                print(output)
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()


    def std_dev_central(users = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used = Average._initialise(users,
                                            loss = loss,
                                            accuracy = accuracy,
                                            post = post,
                                            pre = pre)
        latest_metrics = []
        for user in users.values():
            #print(user.get_latest_accuracy(pre = pre, post = post))
            value = Average._latest_user_metric(user,pre,post,accuracy,loss)
            latest_metrics.append(value)

        latest_metrics = np.asarray(latest_metrics)
        std_dev = latest_metrics.std()
        avg = latest_metrics.mean()


        for user in users.values():
            curr_metric = Average._latest_user_metric(user,pre,post,
                                                    accuracy,loss)
            if curr_metric >= (avg-std_dev):
                user_weights = np.asarray(user.get_weights())
                users_used.add(user.get_id())
                #nested array of [weights] and [biases]
                new_weights += user_weights
            else:
                print(f"User {user.get_id()}: {curr_metric} < {avg-std_dev}")
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()

    def weighted_avg_central(users = None, user = None, weights = None,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used = Average._initialise(users,
                                            loss = loss,
                                            accuracy = accuracy,
                                            post = post,
                                            pre = pre)
        # sum of the accuracies, as thatll be what you divide by, think
        # this is for weighted average division
        acc_sum = 0
        for user in users.values():
            user_weights = np.asarray(user.get_weights())
            users_used.add(user.get_id())

            curr_metric = Average._latest_user_metric(user,pre,post,
                                                    accuracy,loss)
            acc_sum += curr_metric
            # weighted average
            user_weights = user_weights * curr_metric
            # nested array of [weights] and [biases]
            new_weights += user_weights
        new_weights = new_weights/acc_sum
        return new_weights.tolist()
