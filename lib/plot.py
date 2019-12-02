import matplotlib.pyplot as plt
import librosa
import librosa.display


def show_target(name, target):
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


def show_Y(Y):
    show_target("SD", Y[:, 1])
    show_target("HH", Y[:, 0])
    show_target("KD", Y[:, 2])


def show_Y2(Y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(Y[:, 0], color="blue")
    plt.plot(Y[:, 1], color="green")
    plt.plot(Y[:, 2], color="red")

    plt.show()


def show_spec(C, name):
    plt.figure(figsize=(24, 8))
    plt.title(name)
    librosa.display.specshow(C.T, x_axis="time", y_axis="hz", sr=44100, hop_length=512)
    plt.colorbar()
    plt.show()


def show_specs(**kwargs):
    for name, C in kwargs.items():
        show_spec(C, name)


def show_wave(y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(y, color="grey")
    plt.show()


def show_waves(**kwargs):
    for name, y in kwargs.items():
        show_wave(y, name)
