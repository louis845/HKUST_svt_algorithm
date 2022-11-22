import os

os.environ["SCIPY_USE_PROPACK"] = "1"

import numpy as np
import scipy
import scipy.sparse
import random

# create empty sparse matrix with given shape
def create_empty_sparse_matrix(shape):
    return scipy.sparse.csc_matrix(((np.array([]), (np.array([]), np.array([])))), shape = shape)

# create sparse matrix with shape, and given values with their locations
def create_sparse_matrix(shape, values, location_row, location_col):
    return scipy.sparse.csc_matrix(((np.array(values), (np.array(location_row), np.array(location_col)))), shape = shape)

# sample random entries from a height x width matrix.
# returns an array of 2-tuples, each 2-tuple is a location
def generate_locations(width, height, fixed_entries_num):
    locations = []
    perm = np.random.permutation(width*height)
    
    for k in range(fixed_entries_num):
        val = int(perm[k])
        i = int(val/width)
        j = val%width
        locations.append([i, j])
    
    return locations

# converts the locations generated with the above function
# into 2 arrays contains the information of locations
def convert_locations(locations):
    loc = [[],[]]
    for pt in locations:
        loc[0].append(pt[0])
        loc[1].append(pt[1])
    return loc

# from the full matrix M, obtain a sparse matrix with locations 
# this means a new matrix N for which N=M on the locations, but
# equal zero elsewhere.
def filter_locations(M, locations):
    values = []
    for k in range(len(locations[0])):
        values.append(M[locations[0][k],locations[1][k]])

    new_M = create_sparse_matrix([M.shape[0],M.shape[1]], values, locations[0], locations[1])
    return new_M

# generate a random sparse matrix with given shape, and random values on a fixed number of entries.
def generate_matrix(width, height, fixed_entries_num):
    locations = generate_locations(width, height, fixed_entries_num)

    locations = convert_locations(locations)

    values = []
    for k in range(len(locations[0])):
        values.append(random.randint(-100,100)+0.0)

    M = create_sparse_matrix([height,width], values, locations[0], locations[1])
    return M, locations

# generate a random matrix with given rank by taking a product of two matrices.
def generate_matrix_rank(width, height, rank, method="normal", scale = 1.0):
    if method == "normal":
        mat1 = np.random.standard_normal(size = [height, rank])
        mat2 = np.random.standard_normal(size = [rank, width])
    elif method == "uniform":
        mat1 = np.random.uniform(size = [height, rank])
        mat2 = np.random.uniform(size = [rank, width])
    return (mat1 @ mat2) * scale