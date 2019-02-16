from django.shortcuts import render
import json
from django.http import HttpResponse
import numpy as np
from collections import Counter, defaultdict
from django.contrib.staticfiles.templatetags.staticfiles import static
import os

from FindMe.settings import PROJECT_ROOT


def occurrences(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob


def naive_bayes(training, outcome, new_sample):
    classes = np.unique(outcome)
    rows, cols = np.shape(tuple(training))
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)
    #
    class_probabilities = occurrences(outcome)

    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset = training[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:, j])

    for cls in classes:
        for j in range(0, cols):
            likelihoods[cls][j] = occurrences(likelihoods[cls][j])

    results = {}
    for cls in classes:
        class_probability = class_probabilities[cls]
        for i in range(0, len(new_sample)):
            relative_values = likelihoods[cls][i]
            if new_sample[i] in relative_values.keys():
                class_probability *= relative_values[new_sample[i]]
            else:
                class_probability *= 0
            results[cls] = class_probability
    print(results)
    print(results[0])
    print(results[1])
    try:
        ans = str(results[1] / (results[0] + results[1]) * 100)
    except ZeroDivisionError:
        ans = "NULL"
    return ans


# if __name__ == "__main__":
#     myarray = np.genfromtxt(r'training_dataset.csv', delimiter=',', dtype=None)
#     print(myarray)
#     outcome = np.genfromtxt(r'training_outcome.csv', delimiter=',', dtype=None)
#     # training   = np.asarray(((1,0,1,1),(1,1,0,0),(1,0,2,1),(0,1,1,1),(0,0,0,0),(0,1,2,1),(0,1,2,0),(1,1,1,1)));
#     # print(training)
#     # outcome    = np.asarray((0,1,1,1,0,1,0,1))
#     new_sample = np.asarray((3, 5, 1, 1))
#     naive_bayes(myarray, outcome, new_sample)

def index(request):
    q1 = int(request.GET.get('q1', 1))
    q2 = int(request.GET.get('q2', 1))
    q3 = int(request.GET.get('q3', 1))
    q4 = int(request.GET.get('q4', 1))
    print(q1)
    print(q2)
    print(q3)
    print(q4)
    myarray = np.genfromtxt(open(os.path.join(PROJECT_ROOT, 'training_dataset.csv')), delimiter=',', dtype=None)
    print(myarray)
    outcome = np.genfromtxt(open(os.path.join(PROJECT_ROOT, 'training_outcome.csv')), delimiter=',', dtype=None)
    # training   = np.asarray(((1,0,1,1),(1,1,0,0),(1,0,2,1),(0,1,1,1),(0,0,0,0),(0,1,2,1),(0,1,2,0),(1,1,1,1)));
    # print(training)
    # # outcome    = np.asarray((0,1,1,1,0,1,0,1))
    new_sample = np.asarray((q1, q2, q3, q4))
    result = naive_bayes(myarray, outcome, new_sample)
    data = {"data": result}
    json_data = json.dumps(data)
    return HttpResponse(json_data)