import matplotlib.pyplot as plt


def target(name, target):
    plt.figure(figsize=(18, 6))
    plt.title(name)

    if name == "HH":
        color = "blue"
    elif name == "SD":
        color = "green"
    elif name == "KD":
        color = "red"
    else:
        color = "grey"

    plt.plot(target, color=color)
    plt.show()


def Y(Y):
    target("HH", Y[:, 0])
    target("SD", Y[:, 1])
    target("KD", Y[:, 2])


def Y2(Y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(Y[:, 0], color="blue")
    plt.plot(Y[:, 1], color="green")
    plt.plot(Y[:, 2], color="red")

    plt.show()
