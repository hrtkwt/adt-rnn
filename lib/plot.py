import matplotlib.pyplot as plt
import librosa
import librosa.display


def show_target(target, name):
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


def save_target(target, name):
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
    plt.savefig(f'{name}.png')


def show_Y(Y):
    show_target(Y[:, 0], "HH")
    show_target(Y[:, 1], "SD")
    show_target(Y[:, 2], "KD")


def show_Y2(Y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(Y[:, 0], color="blue")
    plt.plot(Y[:, 1], color="green")
    plt.plot(Y[:, 2], color="red")
    plt.show()


def save_Y2(Y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(Y[:, 0], color="blue")
    plt.plot(Y[:, 1], color="green")
    plt.plot(Y[:, 2], color="red")
    plt.savefig(f'{name}.png')


def show_spec(C, name):
    plt.figure(figsize=(24, 8))
    plt.title(name)
    librosa.display.specshow(C.T, x_axis="time", y_axis="hz", sr=44100, hop_length=512)
    plt.colorbar()
    plt.show()


def save_spec(C, name):
    plt.figure(figsize=(24, 8))
    plt.title(name)
    librosa.display.specshow(C.T, x_axis="time", y_axis="hz", sr=44100, hop_length=512)
    plt.colorbar()
    plt.savefig(f'{name}.png')


def show_specs(**kwargs):
    for name, C in kwargs.items():
        show_spec(C, name)


def show_wave(y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(y, color="grey")
    plt.show()


def save_wave(y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(y, color="grey")
    plt.savefig(f'{name}.png')


def show_waves(**kwargs):
    for name, y in kwargs.items():
        show_wave(y, name)
