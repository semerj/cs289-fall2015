import os
import numpy as np
import matplotlib.pylab as plt
from scipy.sparse.linalg import svds
from scipy.misc import imread

def read_images(files, directory, unmasked_pixels):
    img_list = []
    for x in files:
        img = imread(os.path.join(directory, x)).astype(np.uint8)
        unmasked_img = img[unmasked_pixels]
        img_list.append(unmasked_img)
    return np.array(img_list, dtype=np.uint8)

def convert_uint8(og_img):
    img = np.clip(og_img, 0, 255)
    return img.astype(np.uint8)

def display_img(img, mask, unmasked_pixels):
    full_img = np.zeros(np.shape(mask), dtype=np.uint8)
    full_img[unmasked_pixels] = convert_uint8(img)
    plt.axis('off')
    plt.imshow(full_img)

def plot_eigenfaces(faces, face_num, mask, unmasked_pixels):
    face_mean = np.sum(faces, axis=0)/len(faces)
    face_float = faces.astype(float)

    fig = plt.figure(figsize=(12, 4))
    fig_1 = fig.add_subplot(1, 3, 1)
    fig_1.set_title('Original')
    display_img(faces[face_num],
                mask,
                unmasked_pixels)

    fig_2 = fig.add_subplot(1, 3, 2)
    fig_2.set_title('Eigenvalues: 10')
    U, s, V = svds(face_float - face_mean, k=10)
    display_img(U[face_num].dot(np.diag(s).dot(V)) + face_mean,
                mask,
                unmasked_pixels)

    fig_3 = fig.add_subplot(1, 3, 3)
    fig_3.set_title('Error')

    k10_error = np.sum(
        (faces[face_num] - U[face_num].dot(np.diag(s).dot(V)))**2
    )

    U_50, s_50, V_50 = svds(face_float - face_mean, k=50)
    k50_error = np.sum(
        (faces[face_num] - U_50[face_num].dot(np.diag(s_50).dot(V_50)))**2
    )

    U_100, s_100, V_100 = svds(face_float - face_mean, k=100)
    k100_error = np.sum(
        (faces[face_num] - U_100[face_num].dot(np.diag(s_100).dot(V_100)))**2
    )

    pos = [1.5, 2.5, 3.5]
    plt.bar(left=pos,
            height=[k10_error, k50_error, k100_error],
            width=0.5,
            align='center')
    plt.xticks(pos, ['k 10', 'k 50', 'k 100'])


def plot_eigenfeatures(faces, n_eigenfaces, mask, unmasked_pixels):
    face_mean = np.sum(faces, axis=0)/len(faces)
    U, s, V = np.linalg.svd(faces.astype(float) - face_mean,
                            full_matrices=False)
    for num in range(n_eigenfaces):
        fig = plt.figure(figsize=(5, 5))
        eigenface = np.diag(s).dot(V)[num + 1]
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Eigenvalues: {}'.format(num + 1))
        display_img(eigenface + face_mean,
                    mask,
                    unmasked_pixels)
