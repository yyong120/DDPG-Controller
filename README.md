# DDPG-Controller
This project implemented the  deep deterministic policy gradient \(DDPG\) algorithm \(inspired by [the code from Phil](https://www.youtube.com/watch?v=jDll4JSI-xo)\) to train a controller for a chemical process taking place in a continuous stirred tank reactor (CSTR) model.

Please use `requirements.txt` to install all the dependencies.
```shell
pip install -r requirements.txt
```

## Training
`ece228-ddpg-v0.ipynb` is the notebook for training \(I used Kaggle Notebooks, since sometimes colab free version is unstable.\). The trained models are stored in the `checkpoint/` directory, which can be used for simulation and plotting.

## Run the Code
### Run the simulation
```shell
python simu.py
```
This will generate several `.npy` files under the `graphs/` directory, which store the trajectories of system states and control inputs starting from different intial points.

### Plot
```shell
python plot.py
```
This will generate plenty of graphs under the `graphs/` directory, including the noise added during simulation, the trajectories of system states and control inputs starting from different intial points, Q-values after training, accumulated rewards and steps in each episode.

### Compare DDPG controller and model predictive controller
```shell
python perf_comp.py
```
This will show the accuracy and performance of the two controllers.
