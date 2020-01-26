from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf

from user import User

from sklearn.model_selection import train_test_split

def init_model(init_seed=None):
    """
    initialise and return a model
    """
    model = keras.Sequential([
        keras.layers.Flatten(),
#         keras.layers.Dense(4096, activation='relu',
#             kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
#         keras.layers.Dense(1024, activation='relu',
#             kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
#         keras.layers.Dense(128, activation='relu',
#             kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(32, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(6, activation='softmax',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed))
    ])

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model

def init_conv_model(labels,image_shape, init_seed=None):
    """
    initialise and return a model
    """
    model = keras.Sequential([
        keras.layers.Flatten(),
#         keras.layers.Dense(4096, activation='relu',
#             kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
#         keras.layers.Dense(1024, activation='relu',
#             kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(128, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(32, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(8, activation='softmax',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed))
    ])

    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


def init_users(df, averaging_methods, averaging_metric="accuracy", seed=None, test_size=0.2, val_size=0.2):
    """
    Requires the DF to contain a "User" column giving numeric identity to a user
    0 to unique_user_count-1

    Averaging method is a list of methods out of which a random one is selected

    initialise users based on dataframe given and assign random averaging method
    to them based on the list passed in.
    returns a dictionary of users(key: user object) and a global user object
    """
    print("Initialising User instances...")
    users = dict()
    num_users = df["User"].nunique()

    for user_id in range(-1,num_users):

        i = user_id # for global user to get all the data

        if user_id < 0: # for global user with id -1
            user_id = None

        df_val, df_val_class,  df_val_user,\
        df_test, df_test_class, df_test_user,\
        df_train, df_train_class, df_train_user = split_dataframe(df=df,
                                                                  for_user=user_id,
                                                                  seed=seed,
                                                                  val_size=val_size,
                                                                  test_size=test_size)
        user_id = i
        if df_train.shape[0]==0:
            print(f"User {user_id} has no data, no instance created...")
            continue

        model = init_model(init_seed = seed)

        option = np.random.RandomState(seed).randint(0,len(averaging_methods))

        users[user_id] = User(user_id=user_id,
                          model = model,
                          averaging_method = averaging_methods[option],
                          averaging_metric = averaging_metric,
                          train_class = df_train_class.values,
                          train_data = df_train.values,
                          val_class = df_val_class.values,
                          val_data = df_val.values,
                          test_class = df_test_class.values,
                          test_data = df_test.values)

    global_user = users.pop(-1)
    global_user.set_averaging_method(averaging_methods[0])
    print(f"{len(users.keys())} User instances and a global user created!")
    return users, global_user



def split_dataframe(df, for_user=None, val_size=0.2, test_size=0.2, seed=None):
    """
    split the dataframe into train, validation and test splits based on the supplied percentage
    value. The percentage values are relative to the overall dataset size. Same seed is used
    for reproducability.
    Empty dataframes if no data present
    """
    # split into train, validation and test data using sklearn and return dfs for each
    if for_user!=None:
        df = df[df["User"] == for_user]
    if df.shape[0] == 0:
        # if no data for the user, then return 9 empty dfs as per the api
        # print(f"Dataframe for user {user} is of shape {df.shape}, no data. Skipping...")
        df = pd.DataFrame()
        return (df for _ in range(9))


    df_train, df_test = train_test_split(df,
                                         test_size = test_size,
                                         random_state = seed)

    val_size = val_size/(1-test_size)
    df_train, df_val  = train_test_split(df_train,
                                         test_size = val_size,
                                         random_state = seed)

    # store class and user information (in order)
    df_val_class, df_train_class, df_test_class = df_val["Class"], df_train["Class"], df_test["Class"]
    df_val_user,  df_train_user,  df_test_user  = df_val["User"],  df_train["User"],  df_test["User"]

    # drop the class and user identifier columns from data frame
    df_val   = df_val.  drop(df_train.columns[[0,1]], axis=1)
    df_train = df_train.drop(df_train.columns[[0,1]], axis=1)
    df_test  = df_test. drop(df_test. columns[[0,1]], axis=1)
    return df_val, df_val_class,  df_val_user,\
        df_test, df_test_class, df_test_user, \
        df_train, df_train_class, df_train_user


def split_dataframe_tff(df, test_size = 0.2, seed = None):
    
    
    users = list(df["User"].unique())
    
    COL_NAMES = list(df.columns.values)
    output_train = pd.DataFrame(columns = COL_NAMES)
    output_test = pd.DataFrame(columns = COL_NAMES)
    
    for user_id in users:
        user_df = df[df["User"]==user_id]
        
    
        user_df_train, user_df_test = train_test_split(user_df,
                                         test_size = test_size,
                                         random_state = seed)
        
        output_train = output_train.append(user_df_train, ignore_index=True)
        output_test = output_test.append(user_df_test, ignore_index=True)
    return output_train, output_test