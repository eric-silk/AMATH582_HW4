import os
import random  # Holds up spork
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

CROPPED_DIR = "data/CroppedYale"
UNCROPPED_DIR = "data/yalefaces"

def get_all_files(parent_dir):
    x = [i for i in os.walk(parent_dir)]
    print(len(x))
    subdirs = [i for i in x[1:]]
    paths = []
    for subdir in subdirs:
        paths += [os.path.join(subdir[0], i) for i in subdir[2]]
        
    return paths
        
def read_pgm(pgm):
    return(np.asarray(Image.open(pgm)))

def read_flat_pgm(pgm):
    return np.ravel(np.asarray(Image.open(pgm)))

def reconstruction(u, s, vh, modes=None):
    if modes is None:
        modes = s.shape[0]
        
    s_diag = np.zeros((u.shape[0], vh.shape[0]))
    s_diag[:u.shape[1], :u.shape[1]] = np.diag(s)
    
    return np.matmul(np.matmul(u[:,0:modes], s_diag[0:modes, 0:modes]), vh[0:modes, :])

def svd_and_reconstruction(data_matrix, modes=None):
    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)
    return reconstruction(u, s, vh, modes=modes)

def show_reduced_image_plt(reduced_dataset, index):
    img = np.reshape(reduced_dataset[:, index], IMAGE_SHAPE)
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()
    
def show_reduced_image_pil(reduced_dataset, index):
    img = np.reshape(reduced_dataset[:, index], IMAGE_SHAPE)
    tmp = Image.fromarray(img)
    tmp.show()

def mode_subplots(u, s, vh, image_index, mode_list, image_shape):
    mode_list = list(set(mode_list))  # remove dupes
    mode_list = [i for i in mode_list if i > 0]
    mode_list.sort()
    num_images = len(mode_list)
    SUBPLOT_COLS = 2
    subplot_rows = math.ceil(num_images)
    
    fig = plt.figure(figsize=(20,20))
    fig.subplots_adjust(wspace=0.000001)
    #fig.tight_layout()
    for i in range(num_images):
        ax = fig.add_subplot(subplot_rows, SUBPLOT_COLS, i+1)
        ax.title.set_text("Modes: {}".format(mode_list[i]))
        recon = reconstruction(u, s, vh, mode_list[i])
        img = np.reshape(recon[:, image_index], image_shape)
        ax.imshow(img, cmap="gray")

def main():
    cropped = get_all_files(CROPPED_DIR)
    original = len(cropped)
    cropped = [i for i in cropped if "bad" not in i]
    print("{} \"*.bad\" files removed.".format(original - len(cropped)))

    image_shape = read_pgm(cropped[0]).shape

    data_matrix = np.empty((image_shape[0]*image_shape[1], len(cropped)))
    for col, pgm in enumerate(cropped):
        data_matrix[:, col] = read_flat_pgm(pgm)

    # Sanity Check
    i = random.randint(0, len(cropped))
    assert(data_matrix[:, i].all() == read_flat_pgm(cropped[i]).all())

    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)

    modes_to_plot = [1, 10, 100, 1000]
    images_to_plot = [0, 100, 127, 255, 583]
    images_to_plot = [127]

    for i in images_to_plot:
        mode_subplots(u, s, vh, i, modes_to_plot, image_shape)

    plt.show()

if __name__ == "__main__":
    main()
