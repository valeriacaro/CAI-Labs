#!/usr/bin/env python
"""
Simple module implementing LSH
"""

from __future__ import print_function, division
import numpy
import sys
import argparse
import time
from sklearn import datasets

__version__ = '0.3'
__author__ = 'marias@cs.upc.edu'


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' %
              (method.__name__, args, kw, te - ts))
        return result

    return timed

class lsh(object):
    """
    implements lsh for digits database in file 'images.npy'
    """

    def __init__(self, k, m):
        """ k is nr. of bits to hash and m is reapeats """
        # data is numpy ndarray with images
        self.data = datasets.load_digits().images
        self.k = k
        self.m = m

        # determine length of bit representation of images
        # use conversion from natural numbers to unary code for each pixel,
        # so length of each image is imlen = pixels * maxval
        self.pixels = 64
        self.maxval = 16
        self.imlen = self.pixels * self.maxval

        # need to select k random hash functions for each repeat
        # will place these into an m x k numpy array
        numpy.random.seed(12345)
        self.hashbits = numpy.random.randint(self.imlen, size=(m, k))

        # the following stores the hashed images
        # in a python list of m dictionaries (one for each repeat)
        self.hashes = [dict() for _ in range(self.m)]

        # now, fill it out
        self.hash_all_images()

        return

    def hash_all_images(self):
        """ go through all images and store them in hash table(s) """
        # Achtung!
        # Only hashing the first 1500 images, the rest are used for testing
        for idx, im in enumerate(self.data[:1500]):
            for i in range(self.m):
                str = self.hashcode(im, i)

                # store it into the dictionary..
                # (well, the index not the whole array!)
                if str not in self.hashes[i]:
                    self.hashes[i][str] = []
                self.hashes[i][str].append(idx)
        return

    def hashcode(self, im, i):
        """ get the i'th hash code of image im (0 <= i < m)
            notice 'im' is the image itself, *not* the index.
        """
        pixels = im.flatten()
        row = self.hashbits[i]
        str = ""
        for x in row:
            # get bit corresponding to x from image..
            pix = int(x) // int(self.maxval)
            num = x % self.maxval
            if num <= pixels[pix]:
                str += '1'
            else:
                str += '0'
        return str

    def candidates(self, im):
        """ given image im, return set of indices of matching candidates """
        res = set()
        for i in range(self.m):
            code = self.hashcode(im, i)
            if code in self.hashes[i]:
                res.update(self.hashes[i][code])
        return res

def image_distance(im1, im2):
    # Get the shape of the images
    rows, cols = im1.shape

    # Initialize the distance to 0
    distance = 0

    # Iterate through the pixels in the images
    for i in range(rows):
        for j in range(cols):
            # Add the absolute difference between the pixels to the distance
            distance += abs(im1[i, j] - im2[i, j])

    # Return the distance
    return distance

def brute_force_search(query_image, dataset):
    # Initialize variables to store the closest image and the minimum distance
    closest_image = None
    min_distance = float('inf')

    # Iterate through the images in the dataset
    for idx, im in enumerate(dataset):
        # Calculate the distance between the query image and the current image
        distance = image_distance(query_image, im)

        # If the distance is smaller than the current minimum, update the minimum distance and the index of the closest image
        if distance < min_distance:
            min_distance = distance
            closest_image = idx

    # Return the index of the closest image and the minimum distance
    return closest_image, min_distance

def search(query_image, me):
    # Get the candidate set for the query image
    candidates = me.candidates(query_image)

    # If the candidate set is empty, return None
    if not candidates:
        return None

    # Initialize variables to store the closest image and the minimum distance
    closest_image = None
    min_distance = float('inf')

    # Iterate through the images in the candidate set
    for idx in candidates:
        # Get the image from the data array
        im = me.data[idx]

        # Calculate the distance between the query image and the current image
        distance = image_distance(query_image, im)

        # If the distance is smaller than the current minimum, update the minimum distance and the index of the closest image
        if distance < min_distance:
            min_distance = distance
            closest_image = idx

    # Return the index of the closest image and the minimum distance
    return closest_image, min_distance

@timeit
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', default=20, type=int)
    parser.add_argument('-m', default=5, type=int)
    args = parser.parse_args()

    print("Running lsh.py with parameters k =", args.k, "and m =", args.m)

    me = lsh(args.k, args.m)

    # show candidate neighbors for first 10 test images
    for r in range(1500, 1510):
        im = me.data[r]
        cands = me.candidates(im)
        print("there are %4d candidates for image %4d" % (len(cands), r))
        for candidate in cands:
            print(image_distance(im, me.data[candidate]))
        print(brute_force_search(im, me.data[:1500]))
        print(search(im, me))

    return


if __name__ == "__main__":
    sys.exit(main())
