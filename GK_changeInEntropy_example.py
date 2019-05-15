import os, sys, math, random
from itertools import combinations
import numpy as np

# The base denotes what base to use for the log uniform distribution
base = 10

def IQR(data):
    # Determine the interquartile range
    data.sort()
    i = int(len(data) / 4)
    b = data[3*i]
    a = data[i]
    return b-a

def FD(data):
    # Determine bin size via Freedman-Diaconis Method
    return (2*IQR(data))/math.pow(len(data), 1/3.)

def entropy(data):
    minX = min(data)
    maxX = max(data)
    # Bin size is determined via the Freedman-Diaconis Method
    binSize = FD(data)
    if binSize < 0.0001:
        # If the bin size is really small, just declare 0 entropy and move on.
        return 0

    numBins = int(math.ceil((maxX - minX)/binSize))

    # Initialize a histogram
    H = [0.0]*numBins

    for d in data:
        # For each value in the dataset, determine its index in the histogram and add one that element
        idx = 0
        temp = minX + binSize
        while temp < d:
            idx = idx + 1
            temp = temp + binSize
        H[idx] = H[idx] + 1.

    ent = 0
    for i in H:
        if i > 0:
            # If there is a nonzero number in this bin, determine its contribution to the dataset's entropy by first dividing it by the total number of datapoints
            x = i / float(len(data))
            # Then multiply it by its log (divided by the log of 2 to get it into bits)
            ent = ent + x*math.log(x)/math.log(2)
    # Return the entropy
    return -1*ent

def randomUniform(lb, ub):
    # This function takes the lower and upper bounds on a uniform distribution on a log scale and returns the linear number 
    p = random.random()*(ub-lb) + lb

    return math.pow(base, p)

def median(x):
    # Returns the median of dataset x
    n = len(x)
    if n < 1:
        return None
    if n % 2 == 1:
        return sorted(x)[n//2]
    else:
        return sum(sorted(x)[n//2-1:n//2+1])/2.0

def bootstrap(x):
    # Uses bootstrapping (n=1000) to estimate the 95% confidence interval of the median of dataset x
    medians = []
    for i in range(0, 1000):
        y = []
        for i in range(0, len(x)):
            y.append(random.choice(x))
        medians.append(median(y))

    return [sorted(medians)[25], sorted(medians)[975]]


# The values of the "measured" variables as reported in Rowland MA, Fontana WF, Deeds EJ (2012) Biophys J.
avg_on_rate_k  = 1     # 1/(uM s) Kinase-Substrate association rate
avg_off_rate_k = 0.001 # 1/s      Kinase-Substrate dissociation rate
avg_cat_rate_k = 0.999 # 1/s      Kinase-Substrate catalytic rate

avg_Km_k       = (avg_cat_rate_k + avg_off_rate_k) / avg_on_rate_k # uM 
                       #          Michaelis-Menten constant of Kinase for Substrate

avg_on_rate_p  = 1     # 1/(uM s) Phosphatase-Substrate association rate
avg_off_rate_p = 0.001 # 1/s      Phosphatase-Substrate dissociation rate
avg_cat_rate_p = 0.999 # 1/s      Phosphatase-Substrate catalytic rate

avg_Km_p       = (avg_cat_rate_p + avg_off_rate_p) / avg_on_rate_p # uM 
                       #          Michaelis-Menten constant of Phosphatase for Substrate
 
avg_Ko         = 10    # uM       Total concentration of Kinase
avg_Po         = 10    # uM       Total concentration of Phosphatase

avg_So         = 10000 # uM       Total concentration of Substrate

# Building lists to hold the parameter values and their names (for reporting)
params         = [avg_on_rate_k, avg_off_rate_k, avg_cat_rate_k, avg_on_rate_p, avg_off_rate_p, avg_cat_rate_p, avg_Ko, avg_Po, avg_So]
names          = ['on_rate_k', 'off_rate_k', 'cat_rate_k', 'on_rate_p', 'off_rate_p', 'cat_rate_p', 'Ko', 'Po', 'So']

# Build a list of indices for the parameters
indices        = list(range(0, len(params)))

# Build a list of all combinations of length i = 0,...,n parameters
combos = []
for i in range(0, len(params)+1):
    comb = combinations(indices, i)
    for j in list(comb):
        combos.append(j)

# Place to store entropies
entropies = []

# Look at each combination of parameters
for combo in combos:
    # Initialize a list to store model outputs (phosphorylated S)
    Sps = []

    # Run 1000 simulations
    for j in range(0, 1000):
        # List for parameter values for this particular simulation
        random_params = []
        for i in range(0, len(params)):
            if i in combo:
                # This parameter is "measured", so use the "measured" value
                random_params.append(params[i])
            else:
                # This parameter is not measured, so select a randomly chosen value from a uniform random distribution on the logarithmic scale
                random_params.append(randomUniform(math.log(params[i], base)-1, math.log(params[i], base)+1))

        # Assigning parameter values to variables to make it easier to code
        kk    = random_params[0]
        kkn   = random_params[1]
        kkcat = random_params[2]
        
        kp    = random_params[3]
        kpn   = random_params[4]
        kpcat = random_params[5]
        
        Ko    = random_params[6]
        Po    = random_params[7]
        So    = random_params[8]
        
        r     = (Ko*kkcat)/(Po*kpcat) # r is the ratio of the max velocity of the kinase to the max velocity of the phosphatase
        
        Kmk   = (kkn + kkcat) / kk
        Kmp   = (kpn + kpcat) / kp
        
        Kk    = Kmk / So              # Kk and Kp provide ratios of the Michaelis constant to the total substrate concentration, a measure of the saturation of the enzymes (<1 means saturated)
        Kp    = Kmp / So
        
        # Solve for the pseudo-steady state fraction of phosphorylated S
        if r != 1:
            Sp    = (r - 1 - Kk - r*Kp + math.sqrt(math.pow(r - 1 - Kk - r*Kp, 2) + 4*(r-1)*r*Kp))/(2*(r-1))
        else:
            Sp    = Kp / (Kk + Kp)
            
        # Save the fraction of phosphorylated S
        Sps.append(Sp)

    # Determine the Shannon entropy of the distribution of phosphorylated S from the 1000 simulations and save it
    E = entropy(Sps)
    entropies.append(E)

medians = []
results = []

for i in indices:
    # Find all changes in entropy related to the measuring of a parameter. Find all combinations in which the parameter is not already measured, get the entropy associated with that combo. Then get the entropy associated with that combo plus measurement of the parameter
    temp = [] # List to store changes in entropies for the parameter i
    for combo in combos:
        if i not in combo:
            e1 = entropies[combos.index(combo)]                       # This is pre-measuring the parameter of interest
            e2 = entropies[combos.index(tuple(sorted(combo + (i,))))] # This is post-measuring the parameter of interest
            temp.append(e2 - e1)                                      # Record the change in entropies

    CI = bootstrap(temp) # Bootstrap the distribution of changes in entropy
    med = median(temp)   # Save the median

    medians.append(med)               # Save the median
    results.append([i, CI[0], CI[1]]) # and the bootstrapped estimate of the 95% confidence interval of the median


# Save the results. File structure has a parameter for each row, in descending order based on the median change in entropy
# Each row reports the name of the parameter, the median, the distance from the median to the edges of the 95% confidence interval of the median (for plotting purposes), 
# and the actual values of the 95% confidence interval. Any parameter with a 95% confidence interval that does not include 0 has a significant impact
# on the distribution of model outputs and the uncertainty of the model.

oFile = open('diffEntropy_GK_bootstrapped.dat', 'w')
for m in reversed(sorted(medians)):
    result = results[medians.index(m)]
    oFile.write(str(names[result[0]]) + '\t' + str(m) + '\t' + str(m-result[1]) + '\t' + str(result[2]-m) + '\t' + str(result[1]) + '\t' + str(result[2]) + '\n')

oFile.close()
