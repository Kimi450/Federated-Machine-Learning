import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def draw_graphs(user, loss = True, accuracy = True, save_as = None ):
    # this is from the book 74,75
    # history = model.fit(...)
    """
    this function draws the history graph for the user from the most
    recent fit performed on it
    """

    history_dict = user.get_history_metrics()
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict["sparse_categorical_accuracy"]
    val_acc_values = history_dict['val_sparse_categorical_accuracy']
    epochs = range(1, len(loss_values) + 1)
    plt.xlabel('Epochs')

    if loss:
        plt.plot(epochs, acc_values, 'b', label='Training acc')
        plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.ylabel("sparse_categorical_accuracy")
        plt.legend()
        if save_as:
            plt.savefig(f"accuracy_{save_as}")
        plt.show()
        plt.clf()
    if accuracy:
        plt.plot(epochs, loss_values, 'b', label='Training loss')
        plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.ylabel('Loss')
        plt.legend()
        if save_as:
            plt.savefig(f"loss_{save_as}")
        plt.show()
        plt.clf()
        
def _plot_with_fill(df, x_axis, position, metric, color, std_dev_fill, min_max_fill):
    """
    private function used to plot the average line and provide a fill
    based on the fill strategy passed in
    """

    position_label = f"{position}-fit"
    position_df = df[df["Position"]==f"{position}"]
    avg = position_df["Average"]
    plt.plot(x_axis, avg, color, linewidth = 1, label = f"{position_label} {metric}")

    if std_dev_fill:
        std_dev = position_df["Standard Deviation"]
        plt.fill_between(x_axis, 
                         avg - std_dev, 
                         avg + std_dev, 
                         alpha=0.08, color = color)
    elif min_max_fill:
        mini = position_df["Minimum"]
        maxi = position_df["Maximum"]
        plt.fill_between(x_axis,
                         maxi, 
                         avg, 
                         alpha=0.08, color = color)
        plt.fill_between(x_axis,
                         avg, 
                         mini, 
                         alpha=0.08, color = color)

def _userwise_data(user, 
                   ignore_first_n = 0,
                   metric = "accuracy", 
                   post = False, 
                   pre = False):
    """
    private function used to provide average, std_dev, min, max and final values
    of the defined metric and position which are passed in
    """

    user_data = user.get_data(ignore_first_n = ignore_first_n, 
                              metric = metric,
                              pre = pre,
                              post = post)
    avg = np.average(user_data)
    std_dev = np.std(user_data)
    mini = np.amin(user_data)
    maxi = np.amax(user_data)
    final = user_data[-1]
    return (user_data,avg, std_dev, mini, maxi,final)        

def userwise_stats_df(users,ignore_first_n = 0, 
                      metric = "accuracy", 
                      post = False, 
                      pre = False):
    """
    returns a dataframe of data obtained from userwise_data based on position and metrics with
    cols ["Position", "User", "Average", "Standard Deviation", "Minimum", "Maximum", "Final Value"]
    """

    cols = ["Position", "User", "Average", "Standard Deviation", "Minimum", "Maximum", "Final Value"]
    df = pd.DataFrame(columns = cols)
    df_index = 0    
    for i, user in users.items():
        if post:
            user_data,avg, std_dev, mini, maxi,final = \
                _userwise_data(user = user, 
                    ignore_first_n = ignore_first_n, 
                    metric = metric, 
                    post = post)

            df.loc[df_index] = ["Post", i, avg, std_dev, mini, maxi, final]
            df_index +=1
        if pre:
            user_data,avg, std_dev, mini, maxi,final = \
                   _userwise_data(user = user, 
                          ignore_first_n = ignore_first_n, 
                          metric = metric, 
                          pre = pre)       
            df.loc[df_index] = ["Pre", i, avg, std_dev, mini, maxi, final]

            df_index +=1 
    return df

def avg_user_stats(users, std_dev_fill = False, min_max_fill = False,
                   metric = "accuracy", pre = True, post = True,
                   ignore_first_n = 0, save_as = None, final_values = False):
    
    """
    prints graphs based on per user data and optionally returns the final metric 
    values for them as well. Use save_as to save the graph plotted and can ignore_first_n
    round data in the graphin
    """

    
    if metric not in ["accuracy", "loss"] or (std_dev_fill and min_max_fill):
        print("Please select one from accuracy or loss and one or nonefrom std_dev_fill or min_max_fill")
        return None
    
    fill_type = None
    if std_dev_fill:
        fill_type = "std_dev_fill"
    elif min_max_fill:
        fill_type = "min_max_fill"
    
    
    # data collection into df
    df = userwise_stats_df(users = users, 
                           ignore_first_n = ignore_first_n, 
                           metric = metric, 
                           post = post , pre = pre)
            
    user_ids = list(users.keys())
    # plot here and then fill here
    
    if pre:
        if final_values:
            _print_finals(df = df, position = "Pre", metric = metric, save_as = save_as)
            
        _plot_with_fill(df = df, x_axis = user_ids,
                        position = "Pre",
                        metric = metric,
                        color = "r",
                        min_max_fill = min_max_fill, 
                        std_dev_fill = std_dev_fill)
    
    if post:
        if final_values:
            _print_finals(df = df, position = "Post", metric = metric,save_as = save_as)
        
        _plot_with_fill(df = df, x_axis = user_ids,
                        position = "Post",
                        metric = metric,
                        color = "b",
                        min_max_fill = min_max_fill, 
                        std_dev_fill = std_dev_fill)

    
    plt.xlabel("Users")
    plt.ylabel(f"{metric}")
    plt.title(f"Average {metric} per User with fill type: {fill_type}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as)
    plt.show()
    plt.clf()
    return df


def _print_finals(df, position, metric, save_as):
    """
    prints the final averaged value for metric and position defined
    """
    finals =df[df["Position"]==f"{position}"]
    finals = finals[["User", "Final Value"]]
    print(f"Final {metric} for {position}-fit data")
    print(finals)
    if save_as:
        #df.to_csv(f'{save_as}.csv', mode='a', header=False)
        finals.to_csv(f'{save_as}.csv')        
    print(f"Averaged: {finals['Final Value'].mean()}\n")


def _roundwise_data(users, 
                    ignore_first_n = 0, 
                    metric = "accuracy",
                    post = False, 
                    pre = False):
    """
    returns a tuple of data for based on rounds containing all data, rounds, 
    average, standard deviation, min value and max value per 
    """
    user_data = []
    for i, user in users.items():
        user_data_temp = user.get_data(ignore_first_n = ignore_first_n, 
                                       metric = metric, 
                                       post = post, 
                                       pre = pre)
        user_data.append(user_data_temp)
        
    user_data = np.asarray(user_data)
    rounds = len(user_data[0])
    avg = np.average(user_data, axis = 0)
    std_dev = np.std(user_data,  axis = 0)
    mini = np.amin(user_data, axis = 0)
    maxi = np.amax(user_data, axis = 0)
    return (user_data,rounds,avg, std_dev, mini, maxi)        

def roundwise_stats_df(users,ignore_first_n = 0, 
                       metric = "accuracy",
                       post = False, pre = False):
    
    """
    returns a data frame of the cols
    ["Position", "Round", "Average", "Standard Deviation", "Minimum", "Maximum"]
    based on data recieved from roundwise_data catered to position and metric passed in
    """
    
    cols = ["Position", "Round", "Average", "Standard Deviation", "Minimum", "Maximum"]
    df = pd.DataFrame(columns = cols)
        
    # collect user metric values into a numpy array
    # of shape (number of users, number of rounds)
    # and calculate roundwise average across all users returning an
    # array for the average of the round across all users
    if post:
        user_data_post,rounds,post_avg, post_std_dev, post_mini, post_maxi = \
                          _roundwise_data(users = users,
                          ignore_first_n = ignore_first_n, 
                          metric = metric,
                          post = post)
    if pre:
        user_data_pre,rounds,pre_avg, pre_std_dev, pre_mini, pre_maxi = \
                          _roundwise_data(users = users,
                          ignore_first_n = ignore_first_n, 
                          metric = metric,
                          pre = pre)
        
    # the arrays consisting of averages for all rounds, across n users
    # is then put into a dataframe for roundwise stats required for plotting
    
    rounds = [i+ignore_first_n for i in range(rounds)]
    df_index = 0
    for rnd in rounds:
        rnd -= ignore_first_n
        if pre:
            df.loc[df_index] = ["Pre", rnd+ignore_first_n, 
                                pre_avg[rnd], pre_std_dev[rnd], 
                                pre_mini[rnd], pre_maxi[rnd]]
            df_index +=1
        
        if post:
            df.loc[df_index] = ["Post", rnd+ignore_first_n, 
                                post_avg[rnd], post_std_dev[rnd], 
                                post_mini[rnd], post_maxi[rnd]]
            df_index +=1    
    return (df, rounds)


def avg_round_stats(users, std_dev_fill = False, min_max_fill = False,
                    metric = "accuracy", pre = True, post = True,
                    ignore_first_n = 0, save_as = None, final_values = False):
    
    
    """
    prints graphs based on per round data and optionally returns the final metric 
    values for them as well. Use save_as to save the graph plotted and can ignore_first_n
    round data in the graphin
    """

    if metric not in ["accuracy", "loss"] or (std_dev_fill and min_max_fill):
        print("Please select one from accuracy or loss and one or nonefrom std_dev_fill or min_max_fill")
        return None

    
    fill_type = None
    if std_dev_fill:
        fill_type = "std_dev_fill"
    elif min_max_fill:
        fill_type = "min_max_fill"
    
    
    # data collection into df
    df, rounds = roundwise_stats_df(users = users, 
                          ignore_first_n = ignore_first_n, 
                          metric = metric,
                          post = post , pre = pre)
    # plot here and then fill here
    if pre:
        if final_values:
            print(f"Final values for Pre-fit {metric}")
            finals = df[df["Position"]=="Pre"].iloc[-1]
            print(finals)
            if save_as:
                #df.to_csv(f'{save_as}.csv', mode='a', header=False)
                finals.to_csv(f'{save_as}.csv')        


        _plot_with_fill(df = df, x_axis = rounds,
                        position = "Pre",
                        metric = metric,
                        color = "r",
                        min_max_fill = min_max_fill, 
                        std_dev_fill = std_dev_fill)
    
    if post:
        if final_values:
            print(f"Final values for Post-fit {metric}")
            finals = df[df["Position"]=="Post"].iloc[-1]
            print(finals,end="\n\n")
            if save_as:
                #df.to_csv(f'{save_as}.csv', mode='a', header=False)
                finals.to_csv(f'{save_as}.csv')        
    
        _plot_with_fill(df = df, x_axis = rounds,
                        position = "Post",
                        metric = metric,
                        color = "b",
                        min_max_fill = min_max_fill, 
                        std_dev_fill = std_dev_fill)

    
    
    plt.xlabel("Rounds")
    plt.ylabel(f"{metric}")
    plt.title(f"Average {metric} per Round with fill type: {fill_type}")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    if save_as:
        plt.savefig(save_as)
    plt.show()
    plt.clf()
    return df
