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

    def _latest_user_metric(user, pre, post, metric, data_type="test"):
        """
        returns the users metric (acc or loss) on their own model
        using their own test data, based on prefit or postfit of the model
        """
        data = None
        if metric == "accuracy":
            data = user.get_latest_accuracy(pre = pre, post = post, data_type=data_type)
        elif metric == "loss":
            data = user.get_latest_loss(pre = pre, post = post,data_type=data_type)
        else:
            raise Exception("Please select loss or accuracy as metric")
        return data

    def _initialise(user, metric, post, pre):
        """
        create and return an empty set of users used and nested numpy array
        initialised with 0s to act as a holder for new weights
        """
        if type(user) == type(dict()):
            user = list(user.values())[0]

        if pre == post or metric not in ["loss", "accuracy"]:
            raise Exception("Please select one of pre or post and loss or accuracy")

        users_used = set()
        #create a numpy array of 0s
        new_weights = np.asarray(user.get_weights())
        for i in new_weights:
            i[i==i] = 0

        return new_weights, users_used

    def all(users = None,
            user = None, weights = None,
            metric = "accuracy",
            post = False,
            pre = False):
        """
        wrapper for the all averaging method, calls the personalised (p2p)
        version or the central version.
        Both should yield the same results
        central model needs
            users is a list of all the users

        p2p model needs
            user is the current user
            weights is the weights of all the other users' models
        """
        new_weights = None
        if users:
            new_weights = Average.all_central(users = users,
                metric = metric,
                post = post,
                pre = pre)

        elif user and weights:
            new_weights = Average.all_personalised(user = user,
                weights = weights,
                metric = metric,
                post = True,
                pre = False)
        return new_weights

    def all_central(users = None,
            metric = "accuracy",
            post = False,
            pre = False):
        """
        Central version is for all weights to be averaged based on the user's
        performance metric on their own model with their own test data
        """
        new_weights, users_used = Average._initialise(users,
                                            metric = metric,
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
            metric = "accuracy",
            post = False,
            pre = False):
        """
        Personalised version is for all weights to be averaged based on the
        user A's performance metric on other users models using user A's test
        data
        """
        new_weights, users_used = Average._initialise(user,
                                            metric = metric,
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

    def eval_user_on_all(user, weights, metric = "accuracy"):
        """
        returns a dictionary of user_id:metric where metric is
        accuracy or loss.
        it places weights from 'weights' in users current model and
        evaluates it on those custom weights models (recieved from all users)
        and tests on the users own test data, returning the metric values as
        a dictionary
        """
        if metric not in ["loss", "accuracy"]:
            raise Exception("Please select one of loss or accuracy")
        evals = dict.fromkeys(list(weights.keys()))
        original_weights = user.get_weights()
        for user_id,weight in weights.items():
            user.set_weights(weight)
            eval_data = user.evaluate(verbose = False, data_type = "val") # evaluate on train and val data instead of test data to avoid leakage
            if metric == "loss":
                eval_data = eval_data[0]
            elif metric == "accuracy":
                eval_data = eval_data[1]

            evals[user_id] = eval_data

        user.set_weights(original_weights)
        return evals

    def std_dev(users = None,
            user = None, weights = None,
            metric = "accuracy",
            post = False,
            pre = False):

        """
        wrapper for the std_dev averaging method, calls the personalised (p2p)
        version or the central version.
        central model needs
            users is a list of all the users

        p2p model needs
            user is the current user
            weights is the weights of all the other users' models
        """
        new_weights = None
        if users:
            new_weights = Average.std_dev_central(users = users,
                metric = metric,
                post = post,
                pre = pre)

        elif user and weights:
            new_weights = Average.std_dev_personalised(user = user,
                weights = weights,
                metric = metric,
                post = True,
                pre = False)
        return new_weights

    def std_dev_central(users = None,
            metric = "accuracy",
            post = False,
            pre = False):
        """
        Central version is for weights to be averaged based on the user's
        performance metric on their own model with their own test data and
        if the metric is greater than (average of metrics - std dev), then
        the weights are used for averaging, else the weights are discarded.
        """

        new_weights, users_used = Average._initialise(users,
                                            metric = metric,
                                            post = post,
                                            pre = pre)
        latest_metrics = []
        for user in users.values():
            #print(user.get_latest_accuracy(pre = pre, post = post))
            value = Average._latest_user_metric(user,pre,post,metric,data_type="val")
            latest_metrics.append(value)

        latest_metrics = np.asarray(latest_metrics)
        std_dev = latest_metrics.std()
        avg = latest_metrics.mean()

        for user in users.values():
            curr_metric = Average._latest_user_metric(user,pre,post,metric,data_type="val")
            if metric == "loss":
                criteria = curr_metric <= (avg+std_dev)
            else:
                criteria = curr_metric >= (avg-std_dev)

            if criteria:
                user_weights = np.asarray(user.get_weights())
                users_used.add(user.get_id())
                #nested array of [weights] and [biases]
                new_weights += user_weights
            else:
                if metric == "loss":
                    print(f"User {user.get_id()}: {curr_metric} > {avg+std_dev}")
                else:
                    print(f"User {user.get_id()}: {curr_metric} < {avg-std_dev}")
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()

    def std_dev_personalised(user = None, weights = None,
            metric = "accuracy",
            post = False,
            pre = False):

        """
        Personalised version is for weights to be averaged based on the
        user A's performance metric on each of the other users' models
        using user A's test data and if their metric is greater
        than (average of metrics - std dev)m then the weights are used for
        averaging, else the weights are discarded
        """
        new_weights, users_used = Average._initialise(user,
                                            metric = metric,
                                            post = post,
                                            pre = pre)

        evals = Average.eval_user_on_all(user = user, weights = weights,
                                         metric = metric)
        latest_metrics = list(evals.values())
        latest_metrics = np.asarray(latest_metrics)
        std_dev = latest_metrics.std()
        avg = latest_metrics.mean()

        print(f"User {user.get_id()}: -> ")
        for user_id, curr_metric in evals.items():
            if metric == "loss":
                criteria = curr_metric <= (avg+std_dev)
            else:
                criteria = curr_metric >= (avg-std_dev)
            if criteria:
                user_weights = np.asarray(weights[user_id])
                users_used.add(user_id)
                #nested array of [weights] and [biases]
                new_weights += user_weights
            else:
                if metric == "loss":
                    output=f"User {user_id}: {curr_metric} > {avg+std_dev}"
                else:
                    output=f"User {user_id}: {curr_metric} < {avg-std_dev}"
                print(output)
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()

    def weighted_avg(users = None,
            user = None, weights = None,
            metric = "accuracy",
            post = False,
            pre = False):
        """
        wrapper for the weighted averaging method, calls the personalised (p2p)
        version or the central version.
        central model needs
            users is a list of all the users

        p2p model needs
            user is the current user
            weights is the weights of all the other users' models
        """
        new_weights = None
        if users:
            new_weights = Average.weighted_avg_central(users = users,
                metric = metric,
                post = post,
                pre = pre)

        elif user and weights:
            new_weights = Average.weighted_avg_personalised(user = user,
                weights = weights,
                metric = metric,
                post = True,
                pre = False)
        return new_weights

    def weighted_avg_central(users = None,
            metric = "accuracy",
            post = False,
            pre = False):
        """
        Central version is for all weights to be averaged based on the user's
        performance metric on their own model with their own test data by
        multiplying the metric by the weights and averaging using the
        sum of the metrics as the divisor
        """
        new_weights, users_used = Average._initialise(users,
                                            metric = metric,
                                            post = post,
                                            pre = pre)
        # sum of the accuracies, as thatll be what you divide by, think
        # this is for weighted average division
        acc_sum = 0
        for user in users.values():
            user_weights = np.asarray(user.get_weights())
            users_used.add(user.get_id())

            curr_metric = Average._latest_user_metric(user,pre,post, metric,data_type="val")
            if metric == "loss":
                if curr_metric == 0:
                    curr_metric = 10**(-6)
                curr_metric = 1/curr_metric
            acc_sum += curr_metric
            # weighted average
            user_weights = user_weights * curr_metric
            # nested array of [weights] and [biases]
            new_weights += user_weights
        new_weights = new_weights/acc_sum
        return new_weights.tolist()

    def weighted_avg_personalised(user = None, weights = None,
            metric = "accuracy",
            post = False,
            pre = False):
        """
        Personalised version is for all weights to be averaged based on the
        user A's performance metric on other users models using user A's
        test data by multiplying the metric by the weights and averaging
        using the um of the metrics as the divisor
        """
        new_weights, users_used = Average._initialise(user,
                                            metric = metric,
                                            post = post,
                                            pre = pre)
        # sum of the accuracies, as thatll be what you divide by, think
        # this is for weighted average division
        acc_sum = 0
        evals = Average.eval_user_on_all(user = user, weights = weights,
                                         metric = metric)
        for user_id, weight in weights.items():
            user_weights = np.asarray(weight)
            users_used.add(user_id)

            curr_metric = evals[user_id]
            if metric == "loss":
                if curr_metric == 0:
                    curr_metric = 10**(-6)
                curr_metric = 1/curr_metric
            acc_sum += curr_metric
            # weighted average
            user_weights = user_weights * curr_metric
            # nested array of [weights] and [biases]
            new_weights += user_weights
        new_weights = new_weights/acc_sum
        return new_weights.tolist()
