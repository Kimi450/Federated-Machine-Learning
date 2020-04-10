 from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf

from user import User

from sklearn.model_selection import train_test_split

def init_model(init_seed=None, input_shape=(36,)):
    """
    initialise and return a model
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
#         keras.layers.Dense(4096, activation='relu',
#             kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(512, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(128, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(32, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)),
        keras.layers.Dense(5, activation='softmax',
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
    cant be fully reproducible
    https://rampeer.github.io/2019/06/12/keras-pain.html
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape, kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed))),
    model.add(keras.layers.Dense(8, activation='softmax',
            kernel_initializer=keras.initializers.glorot_uniform(seed=init_seed)))

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
            user_id = None # this will cause global user to have no data

        df_val, df_val_class,  df_val_user,\
        df_test, df_test_class, df_test_user,\
        df_train, df_train_class, df_train_user = split_dataframe(df=df,
                                                                  for_user=user_id,
                                                                  seed=seed,
                                                                  val_size=val_size,
                                                                  test_size=test_size)
        user_id = i
        if df_train.shape[0]==0 and user_id>=0:
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
    for user in users.values():
        global_user.add_test_class(user.get_test_class())
        global_user.add_test_data(user.get_test_data())
        global_user.add_val_data(user.get_val_data())
        global_user.add_val_class(user.get_val_class())
        global_user.add_train_data(user.get_train_data())
        global_user.add_train_class(user.get_train_class())
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


def init_users_image(files, averaging_methods, averaging_metric="accuracy", majority_split=0.7, test_size = 0.2, val_size = 0.2, shape=(80,80,3), seed=None, return_global_user=False, split_method="probabilistic"):
    users = {}
    keys = list(files.keys())
    if return_global_user:
        keys += [-1]
    # initialise users
    for class_id in keys:
        model = init_conv_model(keys, shape, seed)
#         model = init_model()

        option = np.random.RandomState(seed).randint(0,len(averaging_methods))
        users[class_id] = User(user_id=class_id,
                  model = model,
                  averaging_method = averaging_methods[option],
                  averaging_metric = averaging_metric,
                  train_class = np.array([]),
                  train_data = np.array([]),
                  val_class = np.array([]),
                  val_data = np.array([]),
                  test_class = np.array([]),
                  test_data = np.array([]))

    # for class ids in keys, we will now create a majority and rest (of the data) split
    if return_global_user:
        global_user = users.pop(-1)
        global_user.set_averaging_method(averaging_methods[0])
        keys = keys[:-1]

    for class_id in keys:
        # uint8 because values were of range 0-255 and this will cover it. Saves memory
        # by not using float32
        images = np.asarray(files[class_id]).astype("uint8")
        # shuffle first pls
        if split_method == "probabilistic":
            majority_data, rest_data_split = probabilistic_split(images,majority_split,len(keys))
        elif split_method == "exact":
            majority_data, rest_data_split = exact_split(images,majority_split,len(keys))
        rest_data_index = 0
        for user_id in keys:
            if user_id == class_id:
                train_data, train_class, test_data, test_class, val_data, val_class = \
                    train_test_val_split(majority_data, class_id, test_size, val_size)
            else:
                raw_data = rest_data_split[rest_data_index]
                train_data, train_class, test_data, test_class, val_data, val_class = \
                    train_test_val_split(raw_data, class_id, test_size, val_size)
                rest_data_index += 1

            users[user_id].add_test_class(test_class)
            users[user_id].add_test_data(test_data)
            users[user_id].add_val_data(val_data)
            users[user_id].add_val_class(val_class)
            users[user_id].add_train_data(train_data)
            users[user_id].add_train_class(train_class)

    if return_global_user:
        for user in users.values():
            global_user.add_test_class(user.get_test_class())
            global_user.add_test_data(user.get_test_data())
            global_user.add_val_data(user.get_val_data())
            global_user.add_val_class(user.get_val_class())
            global_user.add_train_data(user.get_train_data())
            global_user.add_train_class(user.get_train_class())
        return global_user

    return users

def train_test_val_split(np_data, class_id, test_size, val_size):
    test_data, train_data = np.split(np_data, [int(test_size * len(np_data))])
    val_size = val_size/(1-test_size)
    val_data, test_data = np.split(test_data, [int(val_size * len(test_data))])

    train_class = np.full((train_data.shape[0]),class_id)
    test_class = np.full((test_data.shape[0]),class_id)
    val_class = np.full((val_data.shape[0]),class_id)
#     print(f"              {val_class.shape[0] == val_data.shape[0]}")
    return train_data, train_class, test_data, test_class, val_data, val_class


def probabilistic_split(full_data, majority_split, count):
    probability = majority_split
    prob_mask = np.random.sample(len(full_data))<=probability
    majority_data, rest_data = full_data[prob_mask], full_data[np.invert(prob_mask)]

    ratio = len(rest_data)/(count-1)
    rest_data_split, user_data = [], []
    for others in range(count-1-1):
        len_rest_data = len(rest_data)
        if len_rest_data == 0:
            rest_probability = 1
        else:
            rest_probability = ratio/len_rest_data

        prob_mask = np.random.sample(len_rest_data)<=rest_probability
        user_data = rest_data[prob_mask]
        rest_data = rest_data[np.invert(prob_mask)]

        rest_data_split.append(user_data)
    rest_data_split.append(rest_data)
    return majority_data, np.asarray(rest_data_split)

def exact_split(full_data, majority_split, count):
    majority_data, rest_data = np.split(full_data, [int(majority_split * len(full_data))])
    rest_data_split = np.array_split(rest_data,count-1)
    return majority_data, rest_data_split
