"""

Docstring

"""

import numpy as np
import random
from scipy import stats as ss
import matplotlib.pyplot as plt


def main():

    p1 = np.array([1, 1])
    p2 = np.array([4, 4])

    votes = [1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 3, 3, 3]

    # points = np.array([[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]])
    # p = np.array([2.5, 2])
    # outcomes = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

    # knnp = knn_predict(p, points, outcomes, 2)

    # plt.plot(points[:,0], points[:,1], "ro")
    # plt.plot(p[0], p[1], "bo")

    # plt.show()

    n = 20
    (points, outcomes) = generate_synth_data(n)

    plt.figure()
    plt.plot(points[:n, 0], points[:n, 1], 'ro')
    plt.plot(points[n:, 0], points[n:, 1], 'bo')
    plt.show()


def distance(p1, p2):
    """Calculate distance between p1 and p2"""
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))


def majority_vote(votes):
    """
    xxx
    """
    vote_counts = {}
    for v in votes:
        if v in vote_counts:
            vote_counts[v] += 1
        else:
            vote_counts[v] = 1

    winners = []
    max_count = max(vote_counts.values())
    for vote, count in vote_counts.items():
        if count == max_count:
            winners.append(vote)

    # if tie, choose any
    return random.choice(winners)


def majority_vote_short(votes):
    """
    Return the most common element in votes
    """
    mode, count = ss.mode(votes)
    return mode


def find_nearest_neighbors(p, points, k=5):
    """
    return indices of k nearest neighbors to point p in list points
    """
    distances = np.zeros(points.shape[0])

    for i in range(len(distances)):
        distances[i] = distance(p, points[i])

    return np.argsort(distances)[:k]


def knn_predict(p, points, outcomes, k=5):
    """
    find k nearest neighbors
    """
    ind = find_nearest_neighbors(p, points, k)

    return majority_vote(outcomes[ind])


def generate_synth_data(n=50):
    """
    generate bivariate normal points, outcomes, return tuple
    """
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))), axis=0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n)))

    return (points, outcomes)


if __name__ == '__main__':
    main()
