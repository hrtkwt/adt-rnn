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
        color = "black"

    plt.plot(target, color=color)
    plt.show()


def save_target(target, now, title, color):
    plt.figure(figsize=(18, 6))
    plt.title(title)
    plt.plot(target, color=color)
    plt.savefig(f'results/{now}/{title}.png')


def show_Y(Y):
    show_target(Y[:, 0], "HH")
    show_target(Y[:, 1], "SD")
    show_target(Y[:, 2], "KD")


def save_Y(Y, now, name, suffix):
    save_target(Y[:, 0], now, f"{name}_HH_{suffix}", "b")
    save_target(Y[:, 1], now, f"{name}_SD_{suffix}", "g")
    save_target(Y[:, 2], now, f"{name}_KD_{suffix}", "r")


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
    plt.plot(y, color="black")
    plt.show()


def save_wave(y, name):
    plt.figure(figsize=(18, 6))
    plt.title(name)
    plt.plot(y, color="black")
    plt.savefig(f'{name}.png')


def show_waves(**kwargs):
    for name, y in kwargs.items():
        show_wave(y, name)
