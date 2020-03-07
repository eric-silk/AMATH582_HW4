import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io.wavfile as wavfile
import random
import os
import glob
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

MASTER_FS = 44100

# Pick one of the following to Classify
TEST_CASE = "genre_classification"
assert(TEST_CASE in ["band_classification", "same_genre", "genre_classification"])

fs, x = wavfile.read("music/Acoustic/andy_mckee_art_of_motion/Art_Of_Motion.wav")
x.shape

def read_song(fname):
    fs, x = wavfile.read(fname)
    assert(x.shape[1] == 2)
    assert(fs == MASTER_FS)  # If not true gonna have to resample
    
    return x

def random_n_seconds(song, n, fs):
    samples = x.shape[0]
    snip_length = fs*n
    index = random.randint(0, samples-snip_length)
    return song[index:index+snip_length]

def stereo_spectrogram(sample, fs):
    f, t, sxx0 = sig.spectrogram(sample[:, 0], fs=fs)
    _, _, sxx1 = sig.spectrogram(sample[:, 1], fs=fs)
    
    return (f, t, sxx0, sxx1)

def flattened_spectrograms(sample, fs):
    f, t, sxx0, sxx1 = stereo_spectrogram(sample, fs)
    
    return np.concatenate((np.ravel(sxx0), np.ravel(sxx1)))

def svd_of_stacked_spectrograms(sxx0, sxx1):
    data_matrix = np.vstack((sxx0, sxx1)).T
    u, s, vh = np.linalg.svd(data_matrix, full_matrices=False)
    
    return u, s, vh

def reconstruct_n_modes(u, s, vh, modes=None):
    if modes is None:
        modes = s.shape[0]
        
    print("u.shape: {}; vh.shape: {}".format(u.shape, vh.shape))
    s_diag = np.zeros((u.shape[0], vh.shape[0]))
    s_diag[:u.shape[1], :u.shape[1]] = np.diag(s)
    
    return np.matmul(np.matmul(u[:,0:modes], s_diag[0:modes, 0:modes]), vh[0:modes, :]).T

def keep_n_modes(u, s, vh, n):
    return (u[:,:n], s[:n], vh[:n, :])

def process_song(fname, label, num_samples):
    x = read_song(fname)
    samples = []
    full_song = read_song(fname)
    for i in range(num_samples):
        samples.append(random_n_seconds(full_song, 5, MASTER_FS))
        
    return [(sample, label) for sample in samples]

def sample_and_label(folder, label, samples):
    files = glob.glob(folder+"/*.wav")
    labeled_data = []
    for f in files:
        labeled_data = labeled_data + process_song(f, label, samples)
        
    return labeled_data

labeled = []
"band_classification", "same_genre", "genre_classification"
if TEST_CASE == "band_classification":
    labeled += sample_and_label("music/Metal/monuments_the_amanuensis", "Monuments", 30)
    labeled += sample_and_label("music/Acoustic/andy_mckee_art_of_motion", "Andy Mckee", 30)
    # There's only 4 songs, try to get more samples
    labeled += sample_and_label("music/Jazz/herbie_hancock_headhunters", "Herbie Hancock", 50)
elif TEST_CASE == "same_genre":
    labeled += sample_and_label("music/Metal/bongripper_satan_worshipping_doom", "Bongripper", 50)
    labeled += sample_and_label("music/Metal/monolord_vaenir", "Monolord", 30)
    labeled += sample_and_label("music/Metal/monuments_the_amanuensis", "Monuments", 30)
elif TEST_CASE == "genre_classification":
    # Metal
    labeled += sample_and_label("music/Metal/bongripper_satan_worshipping_doom", "Metal", 50)
    labeled += sample_and_label("music/Metal/monolord_vaenir", "Metal", 30)
    labeled += sample_and_label("music/Metal/monuments_the_amanuensis", "Metal", 30)
    # Acoustic
    labeled += sample_and_label("music/Acoustic/andy_mckee_art_of_motion", "Acoustic", 30)
    labeled += sample_and_label("music/Acoustic/antoine_dufour_existence/", "Acoustic", 30)
    labeled += sample_and_label("music/Acoustic/michael_headges_aerial_boundaries/", "Acoustic", 30)
    # Jazz
    labeled += sample_and_label("music/Jazz/herbie_hancock_headhunters", "Jazz", 50)
    labeled += sample_and_label("music/Jazz/pat_metheny_imaginary_day", "Jazz", 30)
    labeled += sample_and_label("music/Jazz/weather_report_heavy_weather", "Jazz", 30)
    
else:
    raise ValueError("Invalid test case specification!")

print(labeled[0][0].size)
initial = len(labeled)
labeled = [i for i in labeled if i[0].size == 441000]
print("Removed {} entries for insufficient size.".format(initial-len(labeled)))

print(len(labeled))

labeled_spectrogram_vectors = [(flattened_spectrograms(i[0], MASTER_FS), i[1]) for i in labeled]
del labeled

#X = np.array([i[0] for i in labeled_spectrogram_vectors])
nom = labeled_spectrogram_vectors[0]
X = np.zeros((len(labeled_spectrogram_vectors), nom[0].size))
for i in range(len(labeled_spectrogram_vectors)):
    try:
        X[i, :] = labeled_spectrogram_vectors[i][0]
    except ValueError:
        print("Failed on entry {}".format(i))

y = np.array([i[1] for i in labeled_spectrogram_vectors])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
del labeled_spectrogram_vectors  # Forcibly free this
del X

# Do dem pee-see-eyyys
pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

print(X_train.shape)

neigh = KNeighborsClassifier(n_neighbors=5)
band_classifier = neigh.fit(X_train, y_train)

predicted = neigh.predict(X_test)
report = metrics.classification_report(y_test, predicted)
print(report)
