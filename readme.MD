# Conway's Game of Life (Neural Network Approach)

This project simulates Conway's Game of Life using a neural network approach with TensorFlow. The simulation runs indefinitely, displaying evolving patterns based on the classic rules of the Game of Life.

## Requirements
Ensure you have the following dependencies installed:

```
matplotlib
numpy
tensorflow
```

You can install them using:
```sh
pip install -r requirements.txt
```

## How to Run
Run the Python script to start the simulation:
```sh
python game_of_life_nn.py
```

The simulation will continue running indefinitely, periodically injecting random noise to prevent static patterns. Close the window to exit the program.

## Features
- Uses a convolutional neural network (CNN) to compute cell neighbors.
- Implements toroidal (wrap-around) boundary conditions.
- Injects random noise every 50 iterations to keep the simulation dynamic.
- Automatically exits when the window is closed.

## License
This project is released under the MIT License.
