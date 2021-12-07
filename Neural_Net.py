import numpy as np
import matplotlib.pyplot as plt

NITER = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))


def activate(x, W, b):
    # activate evaluates the sigmoid function at the given vector x_0.
    x_0 = np.dot(W, x) + b
    vsigmoid = np.vectorize(sigmoid)
    return vsigmoid(x_0)


class neural_net:
    # DATA

    W2 = 0.5 * np.random.normal(0, 1, (2, 2))
    W3 = 0.5 * np.random.normal(0, 1, (3, 2))
    W4 = 0.5 * np.random.normal(0, 1, (2, 3))
    b2 = 0.5 * np.random.normal(0, 1, (2, 1))
    b3 = 0.5 * np.random.normal(0, 1, (3, 1))
    b4 = 0.5 * np.random.normal(0, 1, (2, 1))
    savecost = list()

    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def netbp(self):
        # Forward and Back propagate

        # learning rate
        eta = 0.05
        # number of SG iterations

        # Value of cost function at each iteration
        self.savecost = list()

        for counter in range(NITER):
            # Choose random training point
            k = np.random.randint(1, 10)
            x = np.array([[self.x1[k]], [self.x2[k]]])

            # Forward pass
            a2 = activate(x, self.W2, self.b2)
            a3 = activate(a2, self.W3, self.b3)
            a4 = activate(a3, self.W4, self.b4)

            # Backward pass
            delta4 = a4 * (1 - a4) * (a4 - self.y[:, [k]])
            delta3 = a3 * (1 - a3) * (np.dot(self.W4.T, delta4))
            delta2 = a2 * (1 - a2) * (np.dot(self.W3.T, delta3))

            # Gradient step
            self.W2 = self.W2 - eta * np.dot(delta2, x.T)
            self.W3 = self.W3 - eta * np.dot(delta3, a2.T)
            self.W4 = self.W4 - eta * np.dot(delta4, a3.T)
            self.b2 = self.b2 - eta * delta2
            self.b3 = self.b3 - eta * delta3
            self.b4 = self.b4 - eta * delta4

            # Monitor
            newcost = self.cost(self.W2, self.W3, self.W4, self.b2, self.b3, self.b4)
            self.savecost.append(newcost)

    def cost(self, W2, W3, W4, b2, b3, b4):
        costvec = np.array([np.zeros((10, 1))])
        for i in range(1, 10):
            x = np.array([[self.x1[i]], [self.x2[i]]])
            a2 = activate(x, self.W2, self.b2)
            a3 = activate(a2, self.W3, self.b3)
            a4 = activate(a3, self.W4, self.b4)
            costvec[:, i] = np.linalg.norm(self.y[:, [i]] - a4)
        return np.power(np.linalg.norm(costvec), 2)


def main():
    x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
    x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
    y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
    neural = neural_net(x1, x2, y)
    neural.netbp()

    niter = list(range(NITER))

    X = 'Iteration number'
    Y = 'Value of cost function'
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.plot(niter, neural.savecost, "-b")
    plt.savefig("Cost_vs_Iterations.png")

    # Test column vector
    testVector = np.array([[1/3], [1/2]])

    print("W2:")
    print(neural.W2)
    print("W3:")
    print(neural.W3)
    print("W4:")
    print(neural.W4)
    print("b2:")
    print(neural.b2)
    print("b3:")
    print(neural.b3)
    print("b4:")
    print(neural.b4)

    sigm1 = activate(testVector, neural.W2, neural.b2)
    sigm2 = activate(sigm1, neural.W3, neural.b3)
    F_x = activate(sigm2, neural.W4, neural.b4)

    print("F(x): ")
    print(F_x)

    if F_x[0] > F_x[1]:
        print("Category A")
    elif F_x[0] < F_x[1]:
        print("Category B")
    else:
        print("No conclusion")


if __name__ == "__main__":
    main()
