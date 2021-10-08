from mpi4py import MPI
import shutil
import os
import time
import random
import matplotlib.pyplot as plt
import math
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    cwd = os.getcwd()
    path = os.path.join(cwd, 'Results')

    if os.path.exists(path):
        # removes the path if it exists
        shutil.rmtree(path)

    # creates a new path to Results
    os.mkdir(path)

tic = time.time()

P_SIZE = None
cities = None


def calculate_distance(arr, order):
    A = arr[:, order[1:]]-arr[:, order[0:-1]]
    dist = np.sum(np.sqrt(np.sum(A**2, axis=0)))
    dist += np.sqrt(np.sum((arr[:, order[0]]-arr[:, order[-1]])**2))
    return dist


def swap(arr, i, j):
    arr[[i, j]] = arr[[j, i]]


##########################################
##  GA functions ##
# The following are the functions used in the genetic algorithm


def normalizeFitness(fitness):
    sm = sum(fitness)
    for i in range(P_SIZE):
        fitness[i] = fitness[i]/sm


def calculateFitness(cities, population, fitness):
    for i, order in enumerate(population):
        d = calculate_distance(cities, order)
        fitness[i] = (1/(1+d))


def pickOne(population, prob):
    index = 0
    r = np.random.rand()
    random.shuffle(population)
    while r > 0:
        r -= prob[index]
        index += 1
    index -= 1
    return population[index].copy()


def mutate(order, iters, mutation_rate=1, isBest=False):

    n = order.shape[0]
    if iters % 7 == 0:
        for i in range(mutation_rate):
            j = np.random.randint(0, len(order))
            m = np.random.randint(1, int(n/2))
            k = (j+m) % n
            swap(order, k, j)
    else:
        for i in range(mutation_rate):
            j = np.random.randint(0, len(order))
            m = 1
            k = (j+m) % n
            swap(order, k, j)

    if iters % 12 == 0 and mutation_rate != 0:
        np.random.shuffle(order)
        return

    if (iters % 10 == 0 and isBest) or (iters % 69 == 0 and np.random.randint(0, 100) % 7 == 0):
        d = calculate_distance(cities, order)
        p = np.random.randint(0, n)
        for i in range(p, p+math.ceil(n/2)):
            i = i % n
            if np.random.randint(0, 100) % 2 == 0:
                for j in reversed(range(math.ceil(n/2))):
                    k = (i+j) % n
                    swap(order, k, j)
                    temp = calculate_distance(cities, order)
                    if temp < d:
                        d = temp
                        if iters < 100+int(1000/P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return
                    else:
                        swap(order, k, j)
            else:
                for j in range(math.ceil(n/2)):
                    k = (i+j) % n
                    swap(order, k, j)
                    temp = calculate_distance(cities, order)
                    if temp < d:
                        d = temp
                        if iters < 100+int(1000/P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return
                    else:
                        swap(order, k, j)

    if ((iters+1) % 15 == 0 and isBest and iters % 5 != 0) or (iters < 100 and iters % 5 == 0):
        d = calculate_distance(cities, order)
        p = np.random.randint(0, n)
        for i in range(p, p+n):
            i = i % n
            if np.random.randint(0, 100) % 2 == 0:
                for j in reversed(range(int(n/2))):
                    k = (i+j) % n
                    if k < i:
                        temp = i
                        i = k
                        k = temp
                    order[i:k+1] = np.flip(order[i:k+1])
                    temp = calculate_distance(cities, order)
                    if temp < d:
                        d = temp
                        if iters < 100+int(1000/P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return
                    else:
                        order[i:k+1] = np.flip(order[i:k+1])
            else:
                for j in range(int(n/2)):
                    k = (i+j) % n
                    if k < i:
                        temp = i
                        i = k
                        k = temp
                    order[i:k+1] = np.flip(order[i:k+1])
                    temp = calculate_distance(cities, order)
                    if temp < d:
                        d = temp
                        if (iters < 100+int(1000/P_SIZE) and np.random.randint(0, 100) % 17 == 0):
                            return
                    else:
                        order[i:k+1] = np.flip(order[i:k+1])


def crossOver(order1, order2, iters):

    n = len(order1)
    i1 = np.random.randint(0, n)
    i2 = np.random.randint(0, n)

    order = np.zeros(n)
    if iters % 2 == 0 or iters % 3 == 0:
        order1 = np.flip(order1)

    order[i1:i2+1] = order1[i1:i2+1]
    order[:i1] = -1
    order[i2+1:] = -1

    set_in = set(order)
    j = 0
    for i in range(n):
        if order[i] == -1:
            while order2[j] in set_in:
                j += 1
            order[i] = order2[j]
            j += 1

    return np.array(order, dtype=np.int32)


def bestOrder(arr):
    val = math.inf
    res = None
    for i in range(len(arr)):
        d = calculate_distance(cities, arr[i])
        if d < val:
            val = d
            res = arr[i]

    return res


def nextGeneration(population, fitness, iters, best=None):
    nextGen = []
    n = P_SIZE
    if str(type(best)) != str(type(None)):
        mutate(best, iters, mutation_rate=0, isBest=True)
        nextGen.append(best)
        n -= 1

    for i in range(n):
        m_rate = np.random.randint(1, 3)
        order3 = pickOne(population, fitness)
        mutate(order3, iters, mutation_rate=m_rate)

        order1 = pickOne(population, fitness)
        order2 = pickOne(population, fitness)
        num = np.random.randint(0, 100)
        order = crossOver(order1, order2, iters)
        if iters % 3 == 0:
            mutate(order, iters, mutation_rate=m_rate)

        order_to_add = bestOrder([order, order3])
        # print("Added", order_to_add, calculate_distance(cities, order_to_add))
        nextGen.append(order_to_add)

    return nextGen


def calculateBest(population):
    best = None
    val = math.inf
    p_size = len(population)
    for i in range(p_size):
        d = calculate_distance(cities, population[i])
        if d < val:
            val = d
            best = population[i]
    # print(best, calculate_distance(cities, best))
    return best.copy()


def naturalCalamity(population, best, intensity):
    n = len(population)
    # based on the intensity removes some individuals and replaces them with the fittest.
    # simulates a natural calamity where the unfit individuals fail to survive.
    for i in range(intensity):
        j = np.random.randint(0, n)
        population[j] = best.copy()


##########################################
MAX_ITERS = 100

if rank == 0:
    n = int(input("Enter number of cities\n>>>"))
    P_SIZE = int(input("\nPopulation size\n>>>"))
    MAX_ITERS = int(input("\nMaximum number of iterations\n>>>"))
    flag = int(input(
        "Use data from testCase.csv file? 1 for yes (or) 0 for no (recommended)\n>>>"))
    for i in range(size):
        if i != 0:
            comm.send((n, P_SIZE, MAX_ITERS, flag), dest=i, tag=i)
else:
    n, P_SIZE, MAX_ITERS, flag = comm.recv(source=0, tag=rank)

if flag == 1:
    cities = np.loadtxt(open("testCase.csv", "rb"),
                        delimiter=",").reshape(2, n)
else:
    cities = np.random.uniform(0, 100, (2, n))

if size > 1:
    comm.Bcast(cities, root=0)

optimal_dist = math.inf
iters = 0

population = []
fitness = []


order = np.arange(n)

for i in range(P_SIZE):
    np.random.shuffle(order)
    population.append(order.copy())
    fitness.append(1.0)


plt.style.use('seaborn')

calculateFitness(cities, population, fitness)
normalizeFitness(fitness)
best_ever = None

if rank == 0:
    iter_no = []
    tour_len = []

# Genetic algorithm iterations:

while iters < MAX_ITERS:
    if (iters+1) % 5 == 0 and size > 1:
        # once in 5 generations there is migration of individuals between islands
        random.shuffle(population)
        temp = population[0:P_SIZE//2]
        comm.send(population[0:P_SIZE//2], dest=(rank+1) % size, tag=rank)
        temp = comm.recv(source=(rank-1) % size, tag=(rank-1) % size)
        population[0:P_SIZE//2] = temp

    if (iters+1) % 37 == 0:
        # once in every 37 generations there is a natural calamity which whipes out certain phenotypes
        naturalCalamity(population, best_ever,
                        np.random.randint(1, math.ceil(0.5+P_SIZE/2)))

    population = nextGeneration(population, fitness, iters, best_ever)
    calculateFitness(cities, population, fitness)
    normalizeFitness(fitness)

    best_ever = calculateBest(population)
    d_temp = calculate_distance(cities, best_ever)

    if rank != 0:
        comm.Send(best_ever.astype(np.float), dest=0, tag=rank)
    else:
        temp_ord = np.empty(n, dtype=np.float)
        temp_pop = [best_ever]
        for i in range(size):
            if i != 0:
                comm.Recv(temp_ord, source=i, tag=i)
                temp_pop.append(temp_ord.copy().astype(np.int32))

        best_in_the_world = calculateBest(temp_pop)
        d = calculate_distance(cities, best_in_the_world)

        if d < d_temp:
            print("iter: {}. Best not from island 0".format(iters))

        if optimal_dist > d:
            # the best is plotted only if it is better than the previous best.
            optimal_dist = d
            plt.clf()
            plt.plot(cities[0, best_ever[:]], cities[1, best_ever[:]], '-ro')
            plt.plot([cities[0, best_ever[-1]], cities[0, best_ever[0]]],
                     [cities[1, best_ever[-1]], cities[1, best_ever[0]]], '-r')
            toc = time.time()
            iter_no.append(iters)
            tour_len.append(optimal_dist)
            plt.title("Tour Length: {}, Iteration: {}, Runtime: {}ms".format(
                optimal_dist, iters, (toc-tic)*1000))
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.tight_layout()
            plt.savefig("Results/tour{}.PNG".format(iters))
            plt.clf()

    iters += 1

if rank == 0:
    plt.plot(iter_no, tour_len, '-bx')
    plt.xlabel("Iterations")
    plt.ylabel("Tour length")
    plt.title("Convergence")
    plt.tight_layout()
    plt.savefig("Results/convergence{}.PNG".format(rank))
    plt.clf()
    print("Please find the results obtained in the Results directory created at current working directory.\n")
