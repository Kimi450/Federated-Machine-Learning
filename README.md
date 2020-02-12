chine-Learning
Final Year Project in UCC as a Computer Science student. This is a study based around federated machine learning, with the traditional approach (central server averaging the weights and biases) and a different approach with weighted averages based on characteristics.

## State the given problem in your own words

Federated machine learning is the idea (from Google) of anonymised machine learning (or rather deep learning). It is a way to get a Neural Network trained on everyones data, but without having direct access to everyones data.

Traditionally, a Neural Network would require a lot of data from users to train a model that is fairly accurate. But with the federated approach, the users dont have to share their data with anyone else to obtain a better overall model. Instead, they train a model locally no their own data, and then send the weights and biases of the model (the original user data cannot be recreated with these weights and biases) to a server which then averages them and sends them to you all the users. Because the weight and biases are being sent, instead of the users actual data (like their images), privacy is maintained and essentially a model is trained using anonymised data from several users.

My project is based on implementing the way in which Google does this, and then implementing several more strategies proposed by Derek and comparing their outcome. At a high level, these strategies include discarding the weights of users in certain conditions or using a weighted average of their weights and biases.

## Virtual environment setup
Setup a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [virtualenv](https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-virtualenv-with-Python-3) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) (recommended when not using `tensorflow-federated`) and then install the following packages:

```install tensorflow scipy numpy pandas scikit-learn matplotlib jupyterlab pillow```

Note: Latest version of tensorflow includes `tensorflow-gpu`, but you may need to include it manually

Optionally, include `tensorflow_federated nest_asyncio` in the above statement as well.

## SSH Connection to work machine
More info found [here](https://medium.com/@sankarshan7/how-to-run-jupyter-notebook-in-server-which-is-at-multi-hop-distance-a02bc8e78314)

### Create a 2 hop ssh tunnel to the work machine
```ssh -L <PORT1>:localhost:<PORT2> kts1@csgate.ucc.ie -t ssh -L <PORT2>:localhost:<PORT3> kts1@csg25-05.ucc.ie```

Example:

```ssh -L 6543:localhost:6542 kts1@csgate.ucc.ie -t ssh -L 6542:localhost:6541 kts1@csg25-05.ucc.ie```

### On the work machine, run the following command
```jupyter lab --no-browser --port <PORT3>```

Example:

```jupyter lab --no-browser --port 6541```

### On your machine, go to the browser and run
```localhost:<PORT1>```

Example:

```localhost:6543```

