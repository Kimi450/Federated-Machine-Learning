# Federated-Machine-Learning
Final Year Project in UCC as a Computer Science student. This is a study based around federated machine learning, with the traditional approach (central server averaging the weights and biases) and a different approach with weighted averages based on characteristics.

## State the given problem in your own words

Federated machine learning is the idea (from Google) of anonymised machine learning (or rather deep learning). It is a way to get a Neural Network trained on everyone’s data, but without having direct access to everyone’s data. 

Traditionally, a Neural Network would require a lot of data from users to train a model that is fairly accurate. But with the federated approach, the users dont have to share their data with anyone else to obtain a better overall model. Instead, they train a model locally no their own data, and then send the weights and biases of the model (the original user data cannot be recreated with these weights and biases) to a server which then averages them and sends them to you all the users. Because the weight and biases are being sent, instead of the user’s actual data (like their images), privacy is maintained and essentially a model is trained using anonymised data from several users.

My project is based on implementing the way in which Google does this, and then implementing several more strategies proposed by Derek and comparing their outcome. At a high level, these strategies include discarding the weights of users in certain conditions or using a weighted average of their weights and biases. 

## Virtual environment setup
pip install tensorflow scipy numpy pandas scikit-learn matplotlib jupyterlab

## SSH Connection to work machine
More info found [here](https://medium.com/@sankarshan7/how-to-run-jupyter-notebook-in-server-which-is-at-multi-hop-distance-a02bc8e78314)

### Create a 2 hop ssh tunnel to the work machine
ssh -L 8888:localhost:8889 kts1@csgate.ucc.ie -t ssh -L 8889:localhost:8890 kts1@csg25-05.ucc.ie

### On the work machine, run the following command
jupyter lab --no-browser --port 8890

### On your machine, go to the browser and run
localhost:8888

