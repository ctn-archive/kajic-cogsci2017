# Helper functions for finding the categories a given list of animals belong to
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


def sanitize(string):
    """
    Remove spaces at the beginning and end, as well as any periods
    if there is a space, replace it with an underscore
    """
    return string.strip().replace('.', '').replace(' ', '_').lower()


if __name__ == '__main__':
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, 'animal_data')

    input_file = "animal_categories.txt"
    output_file = "animal_cat_dicts.pkl"

    input_text = open(
        os.path.join(path, input_file), 'r').readlines()

    # Create a dictionary for both ways of mapping
    category_to_animal = {}
    animal_to_category = {}

    for line in input_text:
        # The Category is followed by a ':'
        category, animals_line = line.split(':')
        category = category.lower()

        # Each animal is separated by a ','
        animals = animals_line.split(',')

        # Remove extra spaces and the period at the end
        animals = [sanitize(a) for a in animals]

        category_to_animal[category] = animals

        for animal in animals:
            if animal in animal_to_category.keys():
                animal_to_category[animal].append(category)
            else:
                animal_to_category[animal] = [category]

    print(category_to_animal)
    print(animal_to_category)

    with open(os.path.join(path, output_file), 'wb') as f:
        pickle.dump(category_to_animal, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(animal_to_category, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved:', output_file, 'in', path)
