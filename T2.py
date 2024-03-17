import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import random
import statistics
import math
import seaborn as sns


n = 25


x = np.array([0.63, 1.94, 0.2, 0.43, 2.81, 4.72, 1.13, 
 1.96, 2.88, 3.68, 2.49, 3.09, 2.67, 1.45,
   0.32, 2.25, 1.37, 3.81, 1.68, 
 3.06, 0.53, 2.77, 2.35, 1.84, 1.29])
x = np.sort(x)
print(x)


#moda
print("moda", statistics.mode(x.round(1)), sep = " : ")


#median
print("median",statistics.median(x), sep = " : ")

#scope
print("scope", x[n-1] - x[0], sep = " : ")

#assymmetry coeffiecient

print("assym coeff", sps.skew(x, axis=0, bias=False), sep = " : ")


#-----B)-----
sns.set_theme(rc={'axes.facecolor':'white', 'figure.facecolor':'grey'})
x_line = np.linspace(x[0], x[n-1]+1, 10000)
y = np.array([np.sum(x<i) for i in x_line])/len(x)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x_line, y, color = 'blue')
ax.set_title('Empirical distribution function')
ax.set_ylabel('F(x)')
ax.set_xlabel('x')
ax.set_xticks(np.linspace(0, x_line.max(), 12))
ax.set_yticks(np.linspace(0, y.max(), 12))
ax.grid(which='major', alpha = 0.5, color = 'black') 
plt.show()



k = 1 + math.log2(n)
fig, ax = plt.subplots(figsize=(7, 5))
plt.hist(x, bins = int(k), density=True, histtype='bar', edgecolor='black', color = 'coral') #'bar', 'barstacked', 'step', 'stepfilled'
ax.set_title('Histogramm')
plt.show()


fig, ax = plt.subplots(figsize = (10,5))
bp = sns.boxplot(x, orient='h', boxprops={"facecolor":"coral"},
    medianprops={"color": "black"},)
ax.set_title('Boxplot')
bp.set_xlabel('values')
plt.show()



#x_mean:
x_mean = x.mean()
print("x mean :", x_mean)



#-----C)-----

s = np.sqrt(1/(n-1)*np.sum((x-x_mean)**2))
print(s)

h = 2.344*s/(n**0.2)
print(h)

def q_func(x, h, x0):
    return 3/4*(1-((x-x0)/h)**2)

#interval limits:
a = x[0] - h
print(a)

b = x[n-1] + h
print(b)

values = np.linspace(a,b)
print(values)

y = np.zeros_like(values)

for i in range(len(x)-1):
    a1 = x[i] - h
    b1 = x[i] + h 
    ind_a = np.where(values >= a1)[0][0]
    ind_b = np.where(values <= b1)[0][-1]
    y[ind_a:ind_b] += q_func(values[ind_a : ind_b], h, x[i])


fig, ax = plt.subplots(figsize = (10,5))
ax.set_title('Nuclear estimation of distribution density')
plt.plot(values,y/(n*h), color = 'coral')
plt.show()



fig, ax = plt.subplots(figsize = (10,5))
ax.set_title('Nuclear estimation of distribution density')
sns.displot(x, color='coral')
plt.show()

#---- D)
sps.moment(x,1)


mu = 1
D = 0.04
sigma = math.sqrt(D)
x_th = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x_th, sps.norm.pdf(x_th, mu, sigma), color = 'coral')
plt.show()

#Bootstrap estimation of the density of the arithmetic mean distribution

boot_means = []
for _ in range(1000):
    bootsample = np.random.choice(x, size=n, replace=True)
    boot_means.append(bootsample.mean())

sns.displot(boot_means, bins = n, color = 'coral', stat = 'density')
plt.show()

#e) bootstrap estimation of distribution density coefficient of asymmetry

boot_asym = []
for _ in range(1000):
    bootsample = np.random.choice(x, size=n, replace=True)
    boot_asym.append(sps.skew(bootsample))

be = sns.displot(boot_asym, bins = n, color = 'coral', stat="probability")
plt.show()