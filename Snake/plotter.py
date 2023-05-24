import matplotlib.pyplot as plt
from math import sqrt, floor, ceil

"""
Small file used to plot the results of our agents in order to compare them
"""
with open("Results_of_try10_weights_82_1_1600.h5_over_10000_runs.txt") as file:
    frequency = [int(value)/10000 for value in file.read().strip(']\n').strip('[').split(', ')]
    scores = [i for i in range(200)]
    mean = sum([scores[i]*frequency[i] for i in range(200)])
    std_dev = sqrt(sum([(scores[i]-mean)**2 * frequency[i] for i in range(200)]))
    print(std_dev)
    plt.plot(scores[:80], frequency[:80])
    plt.plot([mean, mean], [-0.001, (frequency[floor(mean)]+frequency[ceil(mean)])/2], color="g")
    plt.plot([mean-std_dev, mean-std_dev], [-0.001, (frequency[floor(mean-std_dev)] + frequency[ceil(mean-std_dev)])/2], color="b")
    plt.plot([mean+std_dev, mean+std_dev], [-0.001, (frequency[floor(mean+std_dev)] + frequency[ceil(mean+std_dev)])/2], color="b")
    plt.plot([mean-2*std_dev, mean-2*std_dev], [-0.001, (frequency[floor(mean-2*std_dev)] + frequency[ceil(mean-2*std_dev)])/2], color="r")
    plt.plot([mean+2*std_dev, mean+2*std_dev], [-0.001, (frequency[floor(mean+2*std_dev)] + frequency[ceil(mean+2*std_dev)])/2], color="r")
    plt.title("Score distribution of try10_weights_82_1_1600.h5 over 10000 runs")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()