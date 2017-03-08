"""
Collection of functions that process a sequence of responses yielding simple
statistics:
    > number of category switches
    > response sequence length
    > mean cluster size
"""

from copy import deepcopy
import numpy as np
import os

from cogsci17_semflu.fan import load_animal_categories

try:
    import cPickle as pickle
except ImportError:
    import pickle

animal_path = os.path.join(
    os.path.dirname(__file__), os.pardir, 'animal_data', 'animal_cat_dicts.pkl')

category_to_animal, animal_to_category = load_animal_categories(animal_path)

#TODO: can also try a heuristic with whatever category 'reaches out the furthest'
def get_category_switches_heuristic(sp_list, time_list):
    """
    Always picks the first biggest category it sees. May not always be 'correct'
    but should be decent
    """
    cat_matrix = build_categorization_matrix(sp_list, animal_to_category,
                                             category_to_animal, unfold_matrix=False)
    sol_matrix = np.zeros(cat_matrix.shape)
    
    num_categories = len(category_to_animal.keys())
    wi = 0
    while wi < len(sp_list):
        longest = 0
        best_category = -1
        best_bounds = [-1,-1]
        for ci in range(num_categories):
            if cat_matrix[ci,wi] == 1:
                bounds = get_bounds(wi, cat_matrix[ci,:])
                length = bounds[1] - bounds[0] + 1
                if length > longest:
                    longest = length
                    best_category = ci
                    best_bounds = bounds
        sol_matrix[best_category, best_bounds[0]:best_bounds[1]+1] = 1
        wi = best_bounds[1]+1
    
    
    # Convert the solution matrix into the correct form of output
    si = 0 # switch index, the final value will be the number of category switches
    time_lists = []
    animal_lists = []
    active_indices = [] # keep track of which category indices are being written
    cs_map = {} # mapping of category index to switch index
    num_categories = len(category_to_animal.keys())
    for wi, (sp, tp) in enumerate(zip(sp_list, time_list)):
        # find all category indices that have a '1' in the solution matrix for the current word
        indices = np.where(sol_matrix[:,wi]==1)[0]
        for i in indices:
            if i in active_indices:
                animal_lists[cs_map[i]].append(sp)
                time_lists[cs_map[i]].append(tp)
            else:
                cs_map[i] = si
                animal_lists.append([sp])
                time_lists.append([tp])
                si += 1
        active_indices = list(indices)

    return time_lists, animal_lists

def build_categorization_matrix(sp_list, animal_to_category, 
                                category_to_animal, unfold_matrix=True):

    # Create an ordered list of all of the categories
    # This will be used to index into the matrix
    categories = category_to_animal.keys()
    categories.sort()

    cat_matrix = np.zeros((len(categories), len(sp_list)))
    for i, sp in enumerate(sp_list):
        for cat in animal_to_category[sp]:
            cat_matrix[categories.index(cat),i] = 1

    # If a category ends and then is repeated later on, extend it to a new row to make
    # processing easier. Its new index will be 'old_index' + '# of repeats'*'len(categories)'
    if unfold_matrix:
        cat_matrix = unfold(cat_matrix)

    return cat_matrix

def unfold(cat_matrix):

    num_categories = cat_matrix.shape[0]
    empty_block = np.zeros(cat_matrix.shape)
    for ci in range(num_categories):
        repeats = -1
        reading_ones = False
        for wi in range(cat_matrix.shape[1]):
            if cat_matrix[ci,wi] == 1:
                if not reading_ones:
                    repeats += 1
                    reading_ones = True
                if repeats > 0:
                    nci = num_categories*repeats+ci
                    if nci >= cat_matrix.shape[0]:
                        # extend the matrix to make room
                        cat_matrix = np.append(cat_matrix, empty_block, axis=0)
                    cat_matrix[ci,wi] = 0
                    cat_matrix[nci,wi] = 1
            elif cat_matrix[ci,wi] == 0 and reading_ones:
                reading_ones = False
    return cat_matrix

def get_category_switches(sp_list, animal_to_category, category_to_animal):
    """
    Returns two lists of lists
    The first contains the category names
    The second contains the actual animals corresponding to the categories
    #TODO: make this comment more helpful
    """
    cat_matrix = build_categorization_matrix(sp_list, animal_to_category,
                                             category_to_animal)

    #print("Starting Matrix")
    #print(cat_matrix)

    # Do some form of optimization to come up with the best category mapping from the data

    # Brute forcing it for now, there should be a better way to do this
    sol_matrix = np.zeros(cat_matrix.shape)
    solutions = cat_mat_sol(0, sol_matrix, cat_matrix)
    """
    print("PRINTING SOLUTIONS")
    for s in solutions:
        print(s)
        print("")

    print("COMPLETE!")
    """
    # Find the best solution from the list
    # The first metric is the least number of category switches (i.e. minimize rows used)
    # If there is a tie, the solution with the most entries in each category wins (i.e. maximize sum(sol_matrix))
    # If there is still a tie, pick one at random?

    sol_score = np.zeros((len(solutions),3))
    best_val = 0
    best_index = -1
    for i, s in enumerate(solutions):
        # Calculate scores in a way that both metrics should be maximized
        sol_score[i,0] = s.shape[0] - np.count_nonzero(np.sum(s, axis=1)) #TODO: pick the right axis
        sol_score[i,1] = np.sum(np.sum(s))
        sol_score[i,2] = sol_score[i,0]*1000000+sol_score[i,1] # lazy combined metric

        if sol_score[i,2] > best_val:
            best_val = sol_score[i,2]
            best_index = i

    best_sol = solutions[best_index]
    #print("Best solution:")
    #print(best_sol)

    # Convert the solution matrix into the correct form of output
    si = 0 # switch index, the final value will be the number of category switches
    category_lists = []
    animal_lists = []
    active_indices = [] # keep track of which category indices are being written
    cs_map = {} # mapping of category index to switch index
    num_categories = len(category_to_animal.keys())
    for wi, sp in enumerate(sp_list):
        # find all category indices that have a '1' in the solution matrix for the current word
        indices = np.where(best_sol[:,wi]==1)[0]
        for i in indices:
            if i in active_indices:
                category_lists[cs_map[i]].append(i%num_categories)
                animal_lists[cs_map[i]].append(sp)
            else:
                cs_map[i] = si
                category_lists.append([i%num_categories])
                animal_lists.append([sp])
                si += 1
        active_indices = list(indices)

    return category_lists, animal_lists

#TODO: make sure nothing is passed by reference, or things could go horribly wrong
def cat_mat_sol(wi, sol_mat, cat_mat):

    if wi > cat_mat.shape[1]:
        return sol_mat
    
    # Find all entries in the column indicated by 'wi' (word index) and do a recursive call 
    # with the contiguous section of the row corresponding with a particular entry added to 
    # the solution matrix, and the new index set to the value after that contiguous section
    ret = []
    for ci in range(cat_mat.shape[0]):
        if cat_mat[ci,wi] == 1:
            bounds = get_bounds(wi, cat_mat[ci,:])
            new_sol_mat = deepcopy(sol_mat)
            new_sol_mat[ci,bounds[0]:bounds[1]+1] = 1
            res = cat_mat_sol(bounds[1]+1, new_sol_mat, cat_mat)
            if type(res) == list:
                # An empty list corresponds to a failed solution, so ignore it
                if len(res) > 0:
                    for r in res:
                        ret.append(r)
            else:
                ret.append(res)

    return ret

# Get the bounds of a contiguous set of ones in an array
#TODO: test this
def get_bounds(i, array):
    upper = -1
    lower = -1
    for j in range(i, len(array)):
        if array[j] == 0:
            upper = j - 1
            break
    if upper == -1:
        upper = len(array)

    for j in reversed(range(0,i)):
        if array[j] == 0:
            lower = j + 1
            break

    if lower == -1:
        lower = 0

    return [lower, upper]
