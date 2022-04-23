from re import S
import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        dot_product = nn.DotProduct(self.get_weights(), x)
        return dot_product

        "*** YOUR CODE HERE ***"

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = nn.as_scalar(self.run(x))
        if score >= 0:
            return 1
        return -1
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """

        while True:
            
            unchanged = True

            for x, y in dataset.iterate_once(1):

                prediction = self.get_prediction(x)


                if prediction != nn.as_scalar(y):
                    print("Error made, updating.")
                    unchanged = False
                    weights = self.get_weights()
                    weights.update(x, nn.as_scalar(y))

                # else do nothing

            # If no weights are updated, we reached 100% training accuracy
            if unchanged:
                return


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here

        "*** YOUR CODE HERE ***"
        # I think this refers to the hyperparameters

        # Hidden layer sizes: between 10 and 400.
        # Batch size: between 1 and the size of the dataset. For Q2 and Q3, we require that total size of the dataset be evenly divisible by the batch size.
        # Learning rate: between 0.001 and 1.0.
        # Number of hidden layers: between 1 and 3.
        self.num_hidden_layers = 2
        self.num_features = 10
        self.batch_size = 4
        self.learning_rate = -0.05
        self.bias1 = nn.Parameter(1, 320)
        self.bias2 = nn.Parameter(1, 1)
        self.epsilon = 0.02
        self.weights1 = nn.Parameter(1, 320)
        self.weights2 = nn.Parameter(320, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # x is a Constant object
        # return a prediction on x
        
        # Input function

        ### Layer 1: Linear transformation ###
        ### Layer 2: Activation function ###
        
        predicted_y1 = nn.AddBias(nn.Linear(x, self.weights1), self.bias1)
        
        predicted_y1 = nn.ReLU(predicted_y1)

        predicted_y2 = nn.AddBias(nn.Linear(predicted_y1, self.weights2), self.bias2)
        
        return predicted_y2

        #two sets of weights & biases

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """

        # Loss function: nn.SquareLoss(a, b)
            # Usage: nn.SquareLoss(a, b)
            # Inputs:
            #     a: a Node with shape (batch_size x dim)
            #     b: a Node with shape (batch_size x dim)
            # Output: a scalar Node (containing a single floating-point number) 
        
        "*** YOUR CODE HERE ***" 
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """

        "*** YOUR CODE HERE ***"
        epsilon = 0.002

        for x, y in dataset.iterate_forever(self.batch_size):


            # get gradients of the loss
            # update biases and weights accordingly
            # compute the loss
            loss = self.get_loss(x, y)


            gw1, gb1, gw2, gb2 = nn.gradients(loss, [self.weights1, self.bias1, self.weights2, self.bias2])

            self.weights1.update(gw1, self.learning_rate)

            self.bias1.update(gb1, self.learning_rate)

            self.weights2.update(gw2, self.learning_rate)

            self.bias2.update(gb2, self.learning_rate)

            if nn.as_scalar(loss) <= epsilon:
                print(nn.as_scalar(loss))
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        ## I think the issue rn is mostly how we're choosing our hyperparameters
        ## Playing around with the batch size, learning rate, and the dimensions of the in-between layers should do the trick

        self.num_hidden_layers = 3
        self.num_features = 10
        self.batch_size = 10
        self.learning_rate = -0.07

        #input = 20*20
        #output = 10
        self.bias1 = nn.Parameter(1, 80)
        self.bias2 = nn.Parameter(1, 10)
        # self.bias3 = nn.Parameter(1, 10)
        self.epsilon = 0.02
        self.weights1 = nn.Parameter(28*28, 80)
        self.weights2 = nn.Parameter(80, 10)
        # self.weights3 = nn.Parameter(16, 10)

    def run(self, x):
        
        """Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        ## Note: I added an extra layer in here just to see if it helped.
        # So far it hasn't done much; the thing is still floating at around 96%

        predicted_y1 = nn.AddBias(nn.Linear(x, self.weights1), self.bias1)
        
        predicted_y1 = nn.ReLU(predicted_y1)

        predicted_y2 = nn.AddBias(nn.Linear(predicted_y1, self.weights2), self.bias2)

        # predicted_y2 = nn.ReLU(predicted_y2)
        
        # predicted_y3 = nn.AddBias(nn.Linear(predicted_y2, self.weights3), self.bias3)

        return predicted_y2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """

        "*** YOUR CODE HERE ***"

        accuracy_wanted = 0.98
        while True:

            for x, y in dataset.iterate_once(self.batch_size):
                # compute the loss
                loss = self.get_loss(x, y)




                # get gradients of the loss
                # update x or weights accordingly idk which

                ## This is the version for the two layers
                gw1, gb1, gw2, gb2 = nn.gradients(loss, [self.weights1, self.bias1, self.weights2, self.bias2])

                ## This is the version for three layers
                # gw1, gb1, gw2, gb2, gw3, gb3 = nn.gradients(loss, [self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3])

                self.weights1.update(gw1, self.learning_rate)

                self.bias1.update(gb1, self.learning_rate)

                self.weights2.update(gw2, self.learning_rate)

                self.bias2.update(gb2, self.learning_rate)

                ## Comment these out if you want to use only two layers
                # self.weights3.update(gw3, self.learning_rate)

                # self.bias3.update(gb3, self.learning_rate)

            validation_accuracy = dataset.get_validation_accuracy()
            
            if validation_accuracy >= accuracy_wanted:
                return

            
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]




        ## Hyperparameters ##

        self.batch_size = 20
        self.hidden_size = 28
        self.learning_rate = -0.07

        ## Other ##

        self.W_x = nn.Parameter(self.num_chars, self.hidden_size)
        self.W_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.weights = nn.Parameter(self.hidden_size, 5)

        # self.W2 = nn.Parameter(self.hidden_size, 5)
        # self.W_hidden2 = nn.Parameter(self.hidden_size, 5)
        # self.bias2 = nn.Parameter(1, 5)
        

        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Case: "cat"
        # xs[0] = "c"
        # xs[1] = "a"
        # xs[2] = "t"

        # Input dimension: batch_size x hidden_size
        # Intermediate dimension: hidden_size x 5
        # Output dimension: batch_size x 5

        # xs = a list of length L
 

        
        ## F initial (first layer) ##


        z = nn.Linear(xs[0], self.W_x)
        
        h = nn.ReLU(z)


        ### Subnetwork (RNN) ###

        for x_value in xs[1:]:
            
            # do f(x, h)
            # f --> the weighted sum of x[i] and the previous h
            x_i = nn.Linear(x_value, self.W_x)
            h_i = nn.Linear(h, self.W_hidden)
            z = nn.Add(x_i, h_i)
            h = nn.ReLU(z)


        ### Second layer ###


        h = nn.Linear(z, self.weights)

        return h # A node with shape (batch_size, 5) containing predicted scores (logits)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        accuracy_wanted = 0.86
        while True:

            for x, y in dataset.iterate_once(self.batch_size):
                # compute the loss
                loss = self.get_loss(x, y)


                # get gradients of the loss
                # update x or weights accordingly idk which

                ## Current version of the gradients

                gwx, gwh, gw = nn.gradients(loss, [self.W_x, self.W_hidden, self.weights])


                ## This is the version for the two layers
                # gw1, gb1, gw2, gb2, h1, h2 = nn.gradients(loss, [self.W1, self.bias1, self.W2, self.bias2, self.W_hidden1, self.W_hidden2])

                ## This is the version for three layers
                # gw1, gb1, gw2, gb2, gw3, gb3 = nn.gradients(loss, [self.weights1, self.bias1, self.weights2, self.bias2, self.weights3, self.bias3])

                self.W_x.update(gwx, self.learning_rate)
                self.W_hidden.update(gwh, self.learning_rate)
                self.weights.update(gw, self.learning_rate)

                # self.W1.update(gw1, self.learning_rate)

                # self.bias1.update(gb1, self.learning_rate)

                # self.W2.update(gw2, self.learning_rate)

                # self.bias2.update(gb2, self.learning_rate)

                # self.W_hidden1.update(h1, self.learning_rate)
                
                # self.W_hidden2.update(h2, self.learning_rate)

                ## Comment these out if you want to use only two layers
                # self.weights3.update(gw3, self.learning_rate)

                # self.bias3.update(gb3, self.learning_rate)

            validation_accuracy = dataset.get_validation_accuracy()
            
            if validation_accuracy >= accuracy_wanted:
                return