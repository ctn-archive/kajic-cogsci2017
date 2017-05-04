"""
Creates pickled files containing list of words and pair-wise association
strengthsfor three different sources of association data:
    - Free norms
    - Word2vec (TODO)
    - Beagle dataset

Pickled files are directly used by the model to run the simulation.
"""

from __future__ import print_function

import fan
import os
import pandas as pd
import numpy as np

path = os.path.join(
    os.path.dirname(__file__), os.pardir, 'association_data')

animals_path = os.path.join(
    os.path.dirname(__file__), os.pardir, 'animal_data',
    'animal_words.txt')

animal_words = [w.upper().strip() for w in open(animals_path, 'r').readlines()]

animal_words.append('ANIMAL')

# remove animals which do not have any associations with other animal words in
# FAN
animal_words.remove('DEVIL')
animal_words.remove('SPONGE')


def create_fan_db():
    words, assoc_db = fan.load_free_association_data(
        os.path.join(path, 'raw_fan'))

    f_name = 'fan_db.pkl'
    fan.save_assoc_db(os.path.join(path, f_name), words, assoc_db)
    am, i2w, w2i = fan.get_assoc_mat(
        words, assoc_db, usewords=animal_words,
        normalize=True)

    fan.save_assoc_mat(path, 'fan_mat', am, i2w, w2i)

    print('Created', f_name, 'in', path)


def create_fanbin_db():
    """
    Binary FAN matrix.
    """
    words, assoc_db = fan.load_free_association_data(
        os.path.join(path, 'raw_fan'))

    adb = []
    for cue, target, _ in assoc_db:
        adb.append((cue, target, 1))

    f_name = 'fanbin_db.pkl'
    fan.save_assoc_db(os.path.join(path, f_name), words, adb)
    am, i2w, w2i = fan.get_assoc_mat(
        words, adb, usewords=animal_words)

    fan.save_assoc_mat(path, 'fanbin_mat', am, i2w, w2i)

    print('Created:', f_name, 'and fanbin_mat', 'in', path)


def create_corpora_db(beagle=False, normalize=True):
    """
    If beagle argument is set creates Beagle association matrix, otherwise
    Google Ngram matrix, contains only words that are also in FAN.
    """

    if beagle:
        name = 'beagle_mat'
        df = pd.read_csv(os.path.join(path, 'beagle.csv'), delimiter=',',
                         skiprows=1, header=None)

        beagle_animals = df[0].apply(func=lambda x: x.upper()).values
        beagle_animals = beagle_animals.tolist()

        w2i = dict(zip(beagle_animals, np.arange(len(df.index))))
        i2w = list(beagle_animals)
        mat = df.drop(0, 1).as_matrix()
        np.fill_diagonal(mat, 0)
    else:
        name = 'ngram_mat'
        mat, i2w, w2i = fan.load_assoc_mat(path, 'google_normalized')

    animal_idx = []
    for animal in animal_words:
        if animal in i2w:
            animal_idx.append(i2w.index(animal))
    animal_idx = np.array(animal_idx)

    i2w_a = [i2w[a] for a in animal_idx]
    mat_a = mat[animal_idx][:, animal_idx]
    w2i_a = {w: i for i, w in enumerate(i2w_a)}

    if normalize:
        row_sums = mat_a.sum(axis=1, keepdims=True)
        mat_a = mat_a/row_sums
        mat_a = np.nan_to_num(mat_a)

    fan.save_assoc_mat(path, name, mat_a, i2w_a, w2i_a)
    print('Created', name, 'dataset in:', path)


if __name__ == "__main__":
    create_fan_db()
    create_fanbin_db()  # FAN data without weight information
    create_corpora_db()
    create_corpora_db(beagle=True)
