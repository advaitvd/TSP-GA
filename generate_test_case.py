import numpy as np

n = int(input("Enter number of cities\n>>>"))
cities = np.random.uniform(0, 100, (2, n))

print(cities)
cities.tofile('testCase.csv', sep=',')
