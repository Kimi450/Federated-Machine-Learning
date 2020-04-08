import h5py
import pandas as pd
import numpy as np
import os
from PIL import Image

def read_file(file):
    """
    return 2d df after imputing with 0s"""

    # read data
    df = pd.read_csv(file)

    # replace the question marks with NaN and then change data type to float 32
    df.replace(["?"],np.nan, inplace=True)
    df = df.astype(np.float32)

    # imputation
    df.fillna(0,inplace=True) # fill nulls with 0
    df = df.iloc[1:] # remove the first 0s
    df["Class"] -= 1
    return df

def shuffle_df(df, seed=None):
    """Shuffle dataframe and reset the index"""
    df = df.take(np.random.RandomState(seed=seed).permutation(df.shape[0]))
    df.reset_index(drop = True, inplace = True)
    
    return df


def _acquire_user_data(df, for_user=None, seed=None):
    """
    return a DataFrame (features) and a Series (target) object for a
    given user "for_user"
    """
    # split into train, validation and test data using sklearn and return dfs for each
    if for_user!=None:
        df = df[df["User"] == for_user]
    if df.shape[0] == 0:
        # if no data for the user, then return 9 empty dfs as per the api
        # print(f"Dataframe for user {user} is of shape {df.shape}, no data. Skipping...")
        df = pd.DataFrame()
        return df, df
    target = df["Class"]

    # drop the class and user identifier columns from data frame
    df   = df.drop(df.columns[[0,1]], axis=1)
    return df, target

def create_hdf5(df,name, seed=None):
    """
    create hdf5 files of structure 
    examples
        userID
            points
            label
    returns the number of clients created
    """
    n = 0
    with h5py.File(name, "w") as f:
        examples = f.create_group("examples")
        u_users = df["User"].unique()
        for user_id in u_users:
            grp = examples.create_group(f"{str(user_id)}")
            user_df, target = _acquire_user_data(df = df, for_user=user_id, seed = 0)
            if user_df.shape[0]==0:
                print(f"User {user_id} has no data, no instance created...")
                continue
            n+=1
            grp.create_dataset('points',data=user_df.values)
            grp.create_dataset('label',data=target.values)
    return n


def load_image(infilename):
    img = Image.open(infilename)
    img = img.resize((80,80))
    data = np.asarray(img)
    return data


    
def save_image(np_array, output_filename) :
    im = Image.fromarray(np_array, "RGB")
    im.save(output_filename)     
    
def map_images(data_dir, verbose = False):
    files = {}
    CLASSES = {}
    VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"] # valid extensions

    for class_id, class_dir in enumerate(os.listdir(data_dir)):
        class_name = class_dir
        class_dir = os.path.join(data_dir, class_dir)
        files[class_id] = []
        CLASSES[class_id] = class_name
        for file in os.listdir(class_dir):

            ext = os.path.splitext(file)[1]
            if ext in VALID_IMAGE_EXTENSIONS:

                np_im = load_image(os.path.join(class_dir,file))
                files[class_id].append(np_im)
#                 save_image(np_im,"i.jpg")
        if verbose:
            print(f"Done: {class_dir}")
    return CLASSES, files
