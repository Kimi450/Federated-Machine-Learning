from average import Average

def train_users(users, epochs,
                new_weights = None,
                train_user_verbose_evaluate = 0,
                train_user_verbose_fit = False,
                verbose = True):
    """
    this method is used to train all users on the passed in epochs value
    """
    
    for user in users.values():
        # if user.get_id() < 0:
        #     continue

        if verbose:
            message = f"User {user.get_id()} being trained on the model..."
            print(message)

        user.train(
            epochs = epochs,
            weights = new_weights, # if none, then wont be updated
            verbose_fit = train_user_verbose_fit,
            verbose_evaluate = train_user_verbose_evaluate
        )

        if verbose:
            message = f"User {user.get_id()} done!"
            print(message)

    return



def train_fed(epochs, rounds, users,
              verbose = True,
              strat = "central",
              train_user_verbose = False,
              train_user_verbose_evaluate = False,
              train_user_verbose_fit = False,
              averaging_method = Average.all,
              averaging_pre = False,
              averaging_post = False,
              averaging_metric = "accuracy"):
    """
    this function trains a federation of users using 'strat' stratergy
    central or personalised
    
    central is where all the users send data to a server and the server
    sends back new weights
    
    personalised is where all the users are sent each others data
    and the user tests how their own test data performs on everyone 
    elses models. Based on their policy, they then decide what way
    to average the data.
    """
    new_weights = None
    
    for i in range(rounds):
        
        # users' weights will not be updated till round i+1
        
        if verbose:
            message = f"{'*'*32} {i:^4} {'*'*32}"
            print(message)

        train_users(users = users, epochs = epochs,
                   new_weights = new_weights,
                   verbose = train_user_verbose,
                   train_user_verbose_evaluate = train_user_verbose_evaluate,
                   train_user_verbose_fit = train_user_verbose_fit)
        if strat == "central":
            # calc new weight and pass it to train users 
            # in next round for the users to update their
            # model and retrain on their local train data
            new_weights = averaging_method(users = users, 
                                  pre = averaging_pre,
                                  post = averaging_post, 
                                  metric = averaging_metric)
    
        elif strat == "personalised":
            new_weights = dict()
            for user in users.values():
                # gather everyones models/weights in a dict
                # and pass it to train users in next round
                new_weights[user.get_id()] = user.get_weights()
            
        if verbose:
            message = f"{'*'*32} {'DONE':^4} {'*'*32}"
            print(message)
    return
