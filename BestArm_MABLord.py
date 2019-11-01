# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:15:18 2019

@author: Clement_X240
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

class gaussian_arm:
    
    def __init__(self, mu, sigma, init_arm=20):
        self.mu = mu
        self.sigma = sigma
        self.data = []
        self.init_arm = init_arm
        
        for i in range(init_arm):
            self.pull()
        
    def pull(self):
        self.data.append(np.random.normal(self.mu, self.sigma))
        
    def size_arm(self):
        return len(self.data)
    
    def get_data(self):
        return self.data
    
    def get_params(self):
        return (self.mu, self.sigma)
    
    def re_init(self):
        self.data = []
        for i in range(self.init_arm):
            self.pull()
    
    

def c(n, x):
    f = np.log(1./x) + 3. * np.log(np.log(1./x)) + 1.5 * np.log(np.log(np.exp(1)*n))
    f /= n
    return np.sqrt(f)



class best_arm_identification:
    
    def __init__(self, arms, epsilon=0, confidence=0.95):
        
        self.arms = arms
        self.epsilon = epsilon
        self.confidence = confidence
        self.p_val = []
        self.p_val_valid = []
        self.init_params()
        self.stopping_time = None
        
    def init_params(self):
        self.mu = np.zeros(len(self.arms))
        self.UCB = np.zeros(len(self.arms))
        self.LCB = np.zeros(len(self.arms))
        self.nb_iter = 1
        self.best_arm = "run the algo before"
        
        for i in range(len(self.arms)):
            self.p_val.append([])
        
    def find_p_val(self, i_arm):
        
        gamma_max = 1
        gamma_min = 1e-5
        
        temp1 = self.mu[i_arm] - c(self.arms[i_arm].size_arm(), gamma_max/(2.*len(self.arms))) > self.mu[0] + c(self.arms[0].size_arm(), gamma_max/2.) + self.epsilon
        temp0 = self.mu[i_arm] - c(self.arms[i_arm].size_arm(), gamma_min/(2.*len(self.arms))) > self.mu[0] + c(self.arms[0].size_arm(), gamma_min/2.) + self.epsilon
       
        if temp1==temp0:
            self.p_val[i_arm].append(1)
        else:
            for i in range(20):
                gamma_mean = (gamma_max + gamma_min)/2.
                temp = self.mu[i_arm] - c(self.arms[i_arm].size_arm(), gamma_mean/(2.*len(self.arms))) <= (self.mu[0] + c(self.arms[0].size_arm(), gamma_mean/2.) + self.epsilon)
                
                if temp==0:
                    gamma_max = gamma_mean
                else:
                    gamma_min = gamma_mean
            self.p_val[i_arm].append(gamma_mean)
    
    def get_valid_p_value(self):
        return self.p_val_valid[-1]
    
    def run(self, stopping_time=None):
        ht = 0
        while True:
            if stopping_time!=None:
                if self.nb_iter>stopping_time:
                    self.best_arm = ht
                    break
            for i in range(len(self.arms)):
                self.mu[i] = np.mean(self.arms[i].get_data())
                self.UCB[i] = self.mu[i] + c(self.arms[i].size_arm(), self.confidence/2.)
                self.LCB[i] = self.mu[i] - c(self.arms[i].size_arm(), self.confidence/(2.*len(self.arms)))
                self.find_p_val(i)
                
            self.p_val_valid.append(np.min(self.p_val))
                
            ht = np.argmax(self.mu)
            temp = self.UCB.copy()
            temp[ht] = np.min(self.UCB) -1
            lt = np.argmax(temp)
            if np.sum(self.LCB[0]>np.delete(self.UCB, 0) -self.epsilon)==len(self.arms)-1:
                self.best_arm = 0
                break
            elif (self.LCB[ht]>self.UCB[lt]-self.epsilon) and (self.LCB[ht]>self.UCB[0]+self.epsilon):
                self.best_arm = ht
                break
        
            if self.epsilon>0:
                temp = self.UCB.copy()
                temp[0] = np.min(self.UCB) -1
                ut = np.argmax(temp)
                for idx in np.unique(np.array([0, ut, ht, lt])):
                    self.arms[idx].pull()
                    
            else:
                self.arms[ht].pull()
                self.arms[lt].pull()
            self.nb_iter = self.nb_iter +1
            
            
            

class MAB_LORD:
    
    def __init__(self, arms, W0=None, alpha=0.05, gamma_sequence=None, max_iter=1000):
        if W0!=None:
            self.W0 = W0
        else:
            self.W0 = np.random.uniform(0, alpha)
            
        self.arms = arms
        self.alpha = alpha
        self.nb_iter = 0
        self.Tj = 2000
        self.max_iter = max_iter
        self.best_arm = best_arm_identification(arms)
        self.R = [0]
        self.alpha_j = []
        self.p_value_valid = []
        
        if gamma_sequence!=None:
            self.gamma_sequence = gamma_sequence
        else:
            gamma_sequence = range(self.max_iter, 1, -1)
            gamma_sequence = np.log(gamma_sequence)
            self.gamma_sequence = gamma_sequence/np.sum(gamma_sequence)
    
        
    
    def run(self):
        
        last_disco = 0
        W_disco = self.W0
        W_j = self.W0
        
        while self.nb_iter<self.max_iter-1:
            alpha_j = self.gamma_sequence[self.nb_iter-last_disco] * W_disco
            W = W_j - alpha_j + self.R[self.nb_iter]*(self.alpha - self.W0)
            for i in range(len(self.arms)):
                self.arms[i].re_init()
            self.best_arm_finder = best_arm_identification(self.arms, confidence=alpha_j)
            self.best_arm_finder.run(stopping_time=self.Tj)
            print(self.nb_iter)
            print(self.best_arm_finder.get_valid_p_value(), alpha_j)
            if self.best_arm_finder.get_valid_p_value() < alpha_j:
                self.R.append(1)
                W_disco = W_j
                last_disco = self.nb_iter
            else:
                self.R.append(0)
            last_disco = np.max([last_disco, self.nb_iter*self.R[-1]])
            
            
            self.nb_iter +=1
            W_j = W
            self.p_value_valid.append(self.best_arm_finder.get_valid_p_value())
            self.alpha_j.append(alpha_j.copy())
                
            
        
      
################################# moyenne éloigné
#################################
            
### epsilon=0
   
mu = [2.7, 3.3, 3.1]
sigma = [1, 1, 1]
arms = []
arms.append(gaussian_arm(mu[0], sigma[0]))
arms.append(gaussian_arm(mu[1], sigma[1]))
arms.append(gaussian_arm(mu[2], sigma[2])) 
      
best_arm_finder = best_arm_identification(arms)
best_arm_finder.run()

print("best_arms: ", best_arm_finder.best_arm)
        
plt.figure()
plt.scatter(np.arange(len(mu)), best_arm_finder.mu, label="estimée")
plt.scatter(np.arange(len(mu)), mu, label="réelle")
error = [best_arm_finder.mu - best_arm_finder.LCB, best_arm_finder.UCB -best_arm_finder.mu]
plt.errorbar(np.arange(len(mu)), best_arm_finder.mu, yerr=error,
             ecolor = 'red', capsize = 10)
plt.legend()
plt.title("Gaussienne moyenne éloignée\n best arm trouvé: {}\n nombre d'itérations:{}".format(best_arm_finder.best_arm, best_arm_finder.nb_iter))
plt.show()

plt.figure()
plt.plot(best_arm_finder.p_val_valid)
plt.title("Evolution P-value valide")
plt.xlabel("nb itérations")
plt.ylabel("p-value valid")
plt.show()




### epsilon=0.7
mu = [2.7, 3.3, 3.1]
sigma = [1, 1, 1]
arms = []
arms.append(gaussian_arm(mu[0], sigma[0]))
arms.append(gaussian_arm(mu[1], sigma[1]))
arms.append(gaussian_arm(mu[2], sigma[2])) 
      
best_arm_finder = best_arm_identification(arms, epsilon=0.8)
best_arm_finder.run()

print("best_arms: ", best_arm_finder.best_arm)
        
plt.figure()
plt.scatter(np.arange(len(mu)), best_arm_finder.mu, label="estimée")
plt.scatter(np.arange(len(mu)), mu, label="réelle")
error = [best_arm_finder.mu - best_arm_finder.LCB, best_arm_finder.UCB -best_arm_finder.mu]
plt.errorbar(np.arange(len(mu)), best_arm_finder.mu, yerr=error,
             ecolor = 'red', capsize = 10)
plt.legend()
plt.title("Gaussienne moyenne éloignée\n best arm trouvé: {}\n nombre d'itérations:{}".format(best_arm_finder.best_arm, best_arm_finder.nb_iter))
plt.show()

plt.figure()
plt.plot(best_arm_finder.p_val_valid)
plt.title("Evolution P-value valide")
plt.xlabel("nb itérations")
plt.ylabel("p-value valid")
plt.show()










################################# moyenne proche
#################################


mu = [3.2, 3.28, 3.25]
sigma = [1, 1, 1]
arms = []
arms.append(gaussian_arm(mu[0], sigma[0]))
arms.append(gaussian_arm(mu[1], sigma[1]))
arms.append(gaussian_arm(mu[2], sigma[2])) 
      
best_arm_finder = best_arm_identification(arms)
best_arm_finder.run()

print("best_arms: ", best_arm_finder.best_arm)
        
plt.figure()
plt.scatter(np.arange(len(mu)), best_arm_finder.mu, label="estimée")
plt.scatter(np.arange(len(mu)), mu, label="réelle")
error = [best_arm_finder.mu - best_arm_finder.LCB, best_arm_finder.UCB -best_arm_finder.mu]
plt.errorbar(np.arange(len(mu)), best_arm_finder.mu, yerr=error,
             ecolor = 'red', capsize = 10)
plt.legend()
plt.title("Gaussienne moyenne proche\n best arm trouvé: {}\n nombre d'itérations:{}".format(best_arm_finder.best_arm, best_arm_finder.nb_iter))
plt.show()

plt.figure()
plt.plot(best_arm_finder.p_val_valid)
plt.title("Evolution P-value valide")
plt.xlabel("nb itérations")
plt.ylabel("p-value valid")
plt.show()



############################# MAB online
#############################
mu = [2, 3]
sigma = [1, 1]
arms = []
arms.append(gaussian_arm(mu[0], sigma[0]))
arms.append(gaussian_arm(mu[1], sigma[1]))

MAB = MAB_LORD(arms=arms, max_iter=100)
MAB.run()

plt.figure()
plt.plot(MAB.alpha_j, label=r'$\alpha_j$')
plt.plot(MAB.p_value_valid, label="p-value valid")
plt.xlabel("nb itération")
plt.legend()
plt.show()



mu = [3,4]
sigma = [1, 1]
arms = []
arms.append(gaussian_arm(mu[0], sigma[0]))
arms.append(gaussian_arm(mu[1], sigma[1]))

MAB = MAB_LORD(arms=arms, max_iter=100)
MAB.run()

plt.figure()
plt.plot(MAB.alpha_j, label=r'$\alpha_j$')
plt.plot(MAB.p_value_valid, label="p-value valid")
plt.xlabel("nb itération")
plt.legend()
plt.show()





mu = [3, 3.18]
sigma = [1, 1]
arms = []
arms.append(gaussian_arm(mu[0], sigma[0]))
arms.append(gaussian_arm(mu[1], sigma[1]))

MAB = MAB_LORD(arms=arms, max_iter=200)
MAB.run()

plt.figure()
plt.plot(np.cumsum(MAB.R))
plt.xlabel("nb itérations")
plt.ylabel("Cumsum Rejet")
plt.show
