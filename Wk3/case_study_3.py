"""

Docstring

"""

import numpy as np
import random
from scipy import stats as ss


def main():

    p1 = np.array([1, 1])
    p2 = np.array([4, 4])

    votes = [1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 3, 3, 3]

    print(majority_vote_short(votes))


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


if __name__ == '__main__':
    main()
