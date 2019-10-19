import numpy as np

class Average:
    def __init__(self):
        """
        this is used to average all the users data using 'method' method
        It returns the new weights as a list of numpy arrrays.
        all, std_dev or weighted_average.

        all is where all the users' weights are just averaged, google policy

        std_dev is where users with metrics 1 std_dev less than the average are discarded

        weighted_avg is the weights*metric/(sum of metrics from all users)
        """
        pass

    def _raise(ex):
        raise ex

    def _initialise(users, loss, accuracy, post, pre):

        if pre == post or loss==accuracy:
            raise Exception("Please select one of pre or post and loss or accuracy")

        latest_user_metric = lambda user, pre, post, accuracy, loss:\
                user.get_latest_accuracy(pre = pre, post = post) if accuracy\
                else (user.get_latest_loss(pre = pre, post = post) if loss else\
                Average._raise(Exception("Please select loss or accuracy as metric")))

        users_used = set()
        #create a numpy array of 0s
        new_weights = np.asarray(users[0].get_weights())
        for i in new_weights:
            i[i==i] = 0

        return new_weights, users_used, latest_user_metric

    def all(users,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used, latest_user_metric = Average._initialise(users,
                                                            loss = loss,
                                                            accuracy = accuracy,
                                                            post = post,
                                                            pre = pre)
        for user in users.values():
            user_weights = np.asarray(user.get_weights())
            users_used.add(user.get_id())
            new_weights += user_weights #nested array of [weights] and [biases]
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()

    def std_dev(users,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used, latest_user_metric = Average._initialise(users,
                                                            loss = loss,
                                                            accuracy = accuracy,
                                                            post = post,
                                                            pre = pre)
        latest_metrics = []
        for user in users.values():
            #print(user.get_latest_accuracy(pre = pre, post = post))
            value = latest_user_metric(user,pre,post,accuracy,loss)
            latest_metrics.append(value)

        latest_metrics = np.asarray(latest_metrics)
        std_dev = latest_metrics.std()
        avg = latest_metrics.mean()


        for user in users.values():
            curr_metric = latest_user_metric(user,pre,post,accuracy,loss)
            if curr_metric >= (avg-std_dev):
                user_weights = np.asarray(user.get_weights())
                users_used.add(user.get_id())
                new_weights += user_weights #nested array of [weights] and [biases]
            else:
                print(f"user {user.get_id()}: {curr_metric} < {avg-std_dev}")
        new_weights = new_weights/len(users_used)
        return new_weights.tolist()

    def weighted_avg(users,
            loss = False,
            accuracy = False,
            post = False,
            pre = False):
        new_weights, users_used, latest_user_metric = Average._initialise(users,
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

            curr_metric = latest_user_metric(user,pre,post,accuracy,loss)
            acc_sum += curr_metric

            user_weights = user_weights * curr_metric # weighted average

            new_weights += user_weights #nested array of [weights] and [biases]
        new_weights = new_weights/acc_sum
        return new_weights.tolist()
