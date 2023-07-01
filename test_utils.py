import numpy as np
import matplotlib.pyplot as plt
import torchvision

def load_dataset():

    mnist = torchvision.datasets.MNIST('./', train=True, download=True)

    X = np.zeros((len(mnist), 784))
    for n in range(len(mnist)):
        X[n] = np.array(mnist[n][0]).reshape(-1)
    return X

def test_weights(w1, w2, input_size, hidden_size):
    
    
    assert w1.shape == (input_size, hidden_size), "Error: Shape of your w1 matrix is not correct."
    assert w1.std() < 0.05, "Error: Standard deviation of your w1 matrix is too high."
    assert -0.2 < w1.mean() < 0.2, "Error: Mean of your w1 matrix needs to be near zero."

    assert w2.shape == (hidden_size, input_size), "Error: Shape of your w2 matrix is not correct."
    assert w2.std() < 0.05, "Error: Standard deviation of your w1 matrix is too high."
    assert -0.2 < w1.mean() < 0.2, "Error: Mean of your w2 matrix needs to be near zero."
    
    print("Weights are initialized properly.")
    

def test_relu(relu):
    test_x = np.arange(-5, 5, 0.1).reshape(10, 10)
    expected_output = np.array(
        [[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
        [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        [2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        [3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
        [4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9]]
        )
    
    try:
        test_output = relu(test_x)
        test_diff = test_output - expected_output
        assert abs(test_diff).max() < 1e-8 
        print("'relu' is implemented properly.")
    except:
        print("Error: Your 'reluimplementation is not correct.")



def test_sigmoid(sigmoid):
    test_x = np.arange(-5, 5, 0.1).reshape(10, 10)
    expected_output = np.array(
        [[0.00669285, 0.00739154, 0.00816257, 0.0090133 , 0.0099518 ,
        0.01098694, 0.01212843, 0.01338692, 0.01477403, 0.0163025 ],
       [0.01798621, 0.01984031, 0.02188127, 0.02412702, 0.02659699,
        0.02931223, 0.03229546, 0.03557119, 0.03916572, 0.04310725],
       [0.04742587, 0.05215356, 0.05732418, 0.06297336, 0.06913842,
        0.07585818, 0.0831727 , 0.09112296, 0.09975049, 0.10909682],
       [0.11920292, 0.13010847, 0.14185106, 0.15446527, 0.16798161,
        0.18242552, 0.19781611, 0.21416502, 0.23147522, 0.24973989],
       [0.26894142, 0.2890505 , 0.31002552, 0.33181223, 0.35434369,
        0.37754067, 0.40131234, 0.42555748, 0.450166  , 0.47502081],
       [0.5       , 0.52497919, 0.549834  , 0.57444252, 0.59868766,
        0.62245933, 0.64565631, 0.66818777, 0.68997448, 0.7109495 ],
       [0.73105858, 0.75026011, 0.76852478, 0.78583498, 0.80218389,
        0.81757448, 0.83201839, 0.84553473, 0.85814894, 0.86989153],
       [0.88079708, 0.89090318, 0.90024951, 0.90887704, 0.9168273 ,
        0.92414182, 0.93086158, 0.93702664, 0.94267582, 0.94784644],
       [0.95257413, 0.95689275, 0.96083428, 0.96442881, 0.96770454,
        0.97068777, 0.97340301, 0.97587298, 0.97811873, 0.98015969],
       [0.98201379, 0.9836975 , 0.98522597, 0.98661308, 0.98787157,
        0.98901306, 0.9900482 , 0.9909867 , 0.99183743, 0.99260846]]
        )
    
    
    try:
        test_output = sigmoid(test_x)
        test_diff = test_output - expected_output
        assert abs(test_diff).max() < 1e-8 
        print("'sigmoid' is implemented properly.")
    except:
        print("Error: Your 'sigmoid' implementation is not correct.")



def test_sigmoid_backward(sigmoid_backward):
    test_arg1 = np.arange(-5, 5, 0.1).reshape(10, 10)
    test_arg2   = np.arange(-5, 5, 0.1).reshape(10, 10)
    expected_output = np.array(
        [[ 1.50000000e+02,  1.41659000e+02,  1.33632000e+02,
         1.25913000e+02,  1.18496000e+02,  1.11375000e+02,
         1.04544000e+02,  9.79970000e+01,  9.17280000e+01,
         8.57310000e+01],
       [ 8.00000000e+01,  7.45290000e+01,  6.93120000e+01,
         6.43430000e+01,  5.96160000e+01,  5.51250000e+01,
         5.08640000e+01,  4.68270000e+01,  4.30080000e+01,
         3.94010000e+01],
       [ 3.60000000e+01,  3.27990000e+01,  2.97920000e+01,
         2.69730000e+01,  2.43360000e+01,  2.18750000e+01,
         1.95840000e+01,  1.74570000e+01,  1.54880000e+01,
         1.36710000e+01],
       [ 1.20000000e+01,  1.04690000e+01,  9.07200000e+00,
         7.80300000e+00,  6.65600000e+00,  5.62500000e+00,
         4.70400000e+00,  3.88700000e+00,  3.16800000e+00,
         2.54100000e+00],
       [ 2.00000000e+00,  1.53900000e+00,  1.15200000e+00,
         8.33000000e-01,  5.76000000e-01,  3.75000000e-01,
         2.24000000e-01,  1.17000000e-01,  4.80000000e-02,
         1.10000000e-02],
       [ 3.15544362e-28,  9.00000000e-03,  3.20000000e-02,
         6.30000000e-02,  9.60000000e-02,  1.25000000e-01,
         1.44000000e-01,  1.47000000e-01,  1.28000000e-01,
         8.10000000e-02],
       [ 2.13162821e-14, -1.21000000e-01, -2.88000000e-01,
        -5.07000000e-01, -7.84000000e-01, -1.12500000e+00,
        -1.53600000e+00, -2.02300000e+00, -2.59200000e+00,
        -3.24900000e+00],
       [-4.00000000e+00, -4.85100000e+00, -5.80800000e+00,
        -6.87700000e+00, -8.06400000e+00, -9.37500000e+00,
        -1.08160000e+01, -1.23930000e+01, -1.41120000e+01,
        -1.59790000e+01],
       [-1.80000000e+01, -2.01810000e+01, -2.25280000e+01,
        -2.50470000e+01, -2.77440000e+01, -3.06250000e+01,
        -3.36960000e+01, -3.69630000e+01, -4.04320000e+01,
        -4.41090000e+01],
       [-4.80000000e+01, -5.21110000e+01, -5.64480000e+01,
        -6.10170000e+01, -6.58240000e+01, -7.08750000e+01,
        -7.61760000e+01, -8.17330000e+01, -8.75520000e+01,
        -9.36390000e+01]])
    
    try:
        
        test_output = sigmoid_backward(test_arg1, test_arg2)
        test_diff = test_output - expected_output
        assert abs(test_diff).max() < 1e-8
        
        print("'sigmoid_backward' is implemented properly.")
        
    except:
        print("Error: Your 'sigmoid_backward' implementation is not correct.")
        

    
def test_mean_squared_error(mean_squared_error):
    
    test_arg1 = np.arange(-5, 5, 0.1).reshape(10, 10)
    test_arg2 = 3 * test_arg1 + np.cos(test_arg1)
    
    expected_output = 33.75615723805222
    
    try:
        test_output = mean_squared_error(test_arg1, test_arg2)
        test_diff = test_output - expected_output
        assert abs(test_diff).max() < 1e-8 
        
        print("'mean_squared_error' is implemented properly.")
        
    except:
        print("Error: Your 'mean_squared_error' implementation is not correct.")
        
    
    
    
    

                               