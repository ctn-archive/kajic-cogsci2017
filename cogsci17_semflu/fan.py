import os
import re

import numpy as np
import pandas as pd
try:
    import cPickle as pickle
except ImportError:
    import pickle

from nengo import spa


def load_free_association_data(path, column='FSG'):
    """Loads the free association data from `path` and saves the list of words
    and the tuple as a single pickle file in the fan_db directory.

    Parameters
    ----------
    path : str
        Path to load data from.
    column : str, optional
        Column in the data files that gives the association strength. Use 'FSG'
        for forward strength and 'BSG' for backward strength.
    Returns
    -------
    A tuple (words, association_database) where words is a set of all occuring
    words (either as cue or target) and association_database is a list of
    associations. Each association is a tuple (cue, target, strength).
    """
    normed_responses = 0
    words = set()
    association_database = []

    for filename in os.listdir(path):
        if not filename.startswith('Cue_Target_Pairs.'):
            continue

        # comment='<' is a hackish way to skip HTML tags
        df = pd.read_csv(
            os.path.join(path, filename), skipinitialspace=True,
            comment='<', encoding='latin1')
        df[column] = pd.to_numeric(df[column])

        df_normed = df[df['NORMED?'] == 'YES']
        normed_responses += len(df_normed)

        # extract norms
        for i, row in df_normed.iterrows():
            cue = sanitize(row['CUE']).upper()
            target = sanitize(row['TARGET']).upper()
            words.add(cue)
            words.add(target)

            strength = row[column]
            association_database.append((cue, target, strength))

    assert len(words) == 5018, "Number words should be 5018."
    assert normed_responses == 63619, \
        "Number of normed responses should be 63619"

    return words, association_database


def load_assoc_db(path):
    """
    Returns the complete list of words in FAN and the association database
    stored in a pickled file (by save_assoc_db).
    """
    with open(os.path.join(path), 'rb') as f:
        words = pickle.load(f)
        assoc_db = pickle.load(f)

    return words, assoc_db


def save_assoc_db(path, words, assoc_db):
    with open(path, 'wb') as f:
        pickle.dump(words, f, protocol=2)
        pickle.dump(assoc_db, f, protocol=2)


def sanitize(w):
    return re.sub(r'\W', '_', w)


def get_assoc_mat(words, association_db, usewords=None, normalize=False):
    """
    Returns association matrix and corresponding mappings between words and
    their ids and the other way around.
        usewords: list of words to be extracted from the database, if None use
            all
        normalize: bool, normalize rows of the matrix at the end
    """
    if usewords is None:
        usewords = list(words)

    w2i = {w: i for i, w in enumerate(usewords)}  # word->id mapping
    i2w = list(usewords)

    assoc_mat = np.zeros((len(w2i), len(w2i)))
    for cue, target, strength in association_db:
        if usewords is words or cue in usewords and target in usewords:
            assoc_mat[w2i[cue], w2i[target]] = strength

    if normalize:
        # assoc_mat /= np.linalg.norm(assoc_mat, axis=1)[:, None]
        row_sums = assoc_mat.sum(axis=1, keepdims=True)
        assoc_mat = assoc_mat/row_sums
        assoc_mat = np.nan_to_num(assoc_mat)

    return assoc_mat, i2w, w2i


def gen_spa_vocab(dimensions, word_list):
    """
    Given dimensionality of semantic pointer and a word list, returns SPA
    vocabulary with those words as semantic pointers.
    """
    vocab = spa.Vocabulary(dimensions)
    words = '+'.join(word_list)
    vocab.parse(words)

    return vocab


def to_vocab_and_assoc_mat_subset(dimensions, words, association_db,
                                  num_words, return_words, word_list):
    # index of 4052 corresponds to 'ANIMAL' and must be included in the vocab
    # exclude animal in the optional returned words list

    if word_list is None:
        word_list = list(words)[:num_words]

    vocab = gen_spa_vocab(dimensions, word_list+['ANIMAL'])
    assoc_mat, _, _ = get_assoc_mat(
        words, association_db, usewords=word_list+['ANIMAL'], normalize=False)

    if return_words:
        return vocab, assoc_mat, word_list
    else:
        return vocab, assoc_mat


def load_animal_categories(path):
    with open(path, 'rb') as f:
        ctoa = pickle.load(f)
        atoc = pickle.load(f)

    return ctoa, atoc


def load_for_model_subset(dimensions, path, num_words=5018,
                          return_words=False, word_list=None):
    """
    Builds a smaller vocab and association matrix using only some of the words
    - If a word list is given, the vocab will contain only those words and the
    word 'animal'
    - If a word list is not given, the first 'num_words' words will be used,
    and 'animal' will be included as well
    """
    words, association_db = load_assoc_db(path)
    return to_vocab_and_assoc_mat_subset(
        dimensions, words, association_db, num_words=num_words,
        return_words=return_words, word_list=word_list)


def save_assoc_mat(path, name, strength_mat, id2word, word2id):
    """Save an association matrix.

    Parameters
    ----------
    path : str
        Output directory.
    name : str
        Filename without extension.
    strength_mat : ndarray
        Association matrix.
    id2word: sequence/dict
        Mapping from index to word.
    word2id: dict
        Mapping from word to index.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    np.save(mat_file, strength_mat)
    with open(map_file, 'wb') as f:
        pickle.dump(id2word, f, protocol=2)
        pickle.dump(word2id, f, protocol=2)


def load_assoc_mat(path, name):
    """Load an association matrix.

    Parameters
    ----------
    path : str
        Input directory
    name : str
        Filename without extension.

    Returns:
    --------
    tuple
        (strength_mat, id2word, word2id) with the matrix of association
        strengths strength_mat, mapping from matrix indices to words id2word,
        and mapping from words to matrix indices.
    """
    mat_file = os.path.join(path, name + '.npy')
    map_file = os.path.join(path, name + '.pkl')

    strength_mat = np.load(mat_file)
    with open(map_file, 'rb') as f:
        id2word = pickle.load(f)
        word2id = pickle.load(f)

    return strength_mat, id2word, word2id
