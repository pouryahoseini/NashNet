# NashNet
A deep neural network to find Nash equilibria of normal-form stage games. The network can be trained for a specific number of players and game shapes (symmetric and asymmetric).

## Description
NashNet is a supervisd deep neural network that is trained with normal-form stage games and their Nash equilibria to predict the equilibria in new games. The network can be trained for a specific game shape. The shape of the game, however, can be symmetric or asymmtric. The games can also have any number of players. The number of generated equilibria can be one or any desired number based on the settings in the training time.

The loss function and the network architecture are designed in a way that if more than one equilibrium is being generared, they do not replicate each other and try to cover all the true nash equilibria of the input game. The loss function is following a max-min strategy and the architecture has multiple heads. Because there can be fewer number of Nash equilibria for a game than the generated ones, to combine possibly redundant predictions, the predictions of the neural network are clustered through the DBSCAN method.

More detailed descriptions are added after the pending papers are published.

## The Code
NashNet is written in Python, by utilizing the Tensorflow, Keras, and Scikit-learn libraries. To generate the datasets, the [Gambit](http://www.gambit-project.org) library is used to find the true output values (Nash equilibria) of the regressor network.

The source code can be found under the *src* directory. To start the training and subsequent testing, run the *Run.py*. It uses the configurations stored in the *Config* folder. The trained models are stored in the *Model directory*, with the model snapshots during the training inside the *Model/Interim* folder. At the end of the training and testing, the respective report files are saved in the *Reports* folder.

The file *run.sh* is designed to run the *Run.py* multple times with different training and testing configurations, for which they are stored under the *Configs* directory.  When running the *run.sh*, the trained models and final test and training reports, with their directory structure, are all saved under the *Results* folder.

# Developers
[Pourya Hoseini](https://github.com/pouryahoseini), [Dustin Barnes](https://github.com/brokndremes), and [Tapadhir Das](https://github.com/dastapadhir)

# License
Copyright 2019 - 2020, Pourya Hoseini, Dustin Barnes, Tapadhir Das, and the NashNet contributors. Any usage must be with the permission of the authors.

# Contact
We can be reached at the following email addresses:
- Pourya Hoseini: [hoseini@nevada.unr.edu](mailto:hoseini@nevada.unr.edu)
- Dustin Barnes: [dkbarnes@nevada.unr.edu](dkbarnes@nevada.unr.edu)
- Tapadhir Das: [tapadhird@nevada.unr.edu](tapadhird@nevada.unr.edu)