import math
import numpy as np
import matplotlib.pyplot as plt


def g(x):
    return -x*math.log(x)

def Hdisp(c,m):
    # Compute the entropy associated with the dispersed phase
    return (-(1-(m-1)*c)*math.log(1-(m-1)*c))+((m-1)*c*math.log(1/c))

def Hcond(c,k,S):
    # Compute the entropy associated with the condensed phase
    S_lim = k * (1 + np.exp((-g(c) - g(1 - c)) / c))
    if S<k:
        return math.log(S)
    elif S >= k and S <= S_lim:
        return math.log(k)
    else:
        return (-(1-c)*math.log((1-c)/k))-(c*math.log(c/(S-k)))

def phase_detection(weights, k_max, percent_disp, percent_cond):
   # Store phase classifications and intermediate quantities
   results_distrib = {}
   results_m = {}
   results_M = {}
   results_S = {}
   results_delta = {}
   # Initialize best solution
   best_c = None
   best_k = None
   best_delta = 0
   best_m = None
   best_S = None
   best_score = -np.inf

   for k in range(1, k_max+1):
      # Define the possible threshold values
      if k==1:
        c_values = np.arange(0.001, 0.5, 0.001)
      else:
        c_values = np.arange(0.001, 1/k, 0.001)
      for c in c_values:
        m_dynamic = []
        M_dynamic = []
        S_dynamic = []
        # Compute m, M and S for every time step
        for weights_i in weights:
            sorted_weights_i = np.sort(weights_i)[::-1]
            # Number of weights above threshold c
            m_i = np.sum(sorted_weights_i>=c)
            m_dynamic.append(m_i)
            # Sum of the k largest weights
            M_i = np.sum(sorted_weights_i[:k])
            M_dynamic.append(M_i)
            # Number of persistence weights
            S_i = len(sorted_weights_i)
            S_dynamic.append(S_i)
        # Store temporal quantities
        results_m[(c,k)] = np.array(m_dynamic)
        results_M[(c,k)] = np.array(M_dynamic)
        results_S[(c,k)] = np.array(S_dynamic)
        for m in range(k+1, max(m_dynamic)+1):
            distrib_phases=[]
            for i in range(len(m_dynamic)):
                # Dispersed phase
                if m_dynamic[i] >= m:
                   distrib_phases.append(-1)
                # Condensed phase
                elif (M_dynamic[i] > (1 - c)) and (S_dynamic[i] <= k or M_dynamic[i] >= (k/S_dynamic[i])):
                   distrib_phases.append(1)   # Fase Condensada
                else:
                   distrib_phases.append(0)
            results_distrib[(c,k,m)] = np.array(distrib_phases)
            # Identify time steps belonging to each phase
            disp_phase = np.where(results_distrib[(c,k,m)]==-1)[0]
            cond_phase = np.where(results_distrib[(c,k,m)]==1)[0]
            all_phases = len(results_distrib[(c,k,m)])
            # Check whether both phases have sufficient coverage
            if (len(disp_phase)/all_phases)> percent_disp and (len(cond_phase)/all_phases) > percent_cond:
                coverage = (len(disp_phase)+len(cond_phase))/all_phases
                # Maximum system size in the condensed phase
                S = np.max(np.array(S_dynamic)[cond_phase])
                S_max = np.max(np.array(S_dynamic))
                try:
                   # Compute the entropy gap
                   delta = Hdisp(c, m) - Hcond(c, k, S)
                   results_delta[(c,k,m)]=delta
            
                   if delta > 0:
        
                      # Normalize the entropy gap
                      delta_norm = delta/math.log(S_max)
                                                            
                      score = delta_norm*coverage
                
                      if score > best_score:
                         best_score = score
                         best_c = c
                         best_k = k
                         best_delta = delta
                         best_m = m
                         best_S = S
                except:
                    continue

   return best_c, best_m, best_k, best_S, best_delta, best_score, results_distrib, results_m, results_M


def plot_phases(c, k, m_bar, results_distrib, results_m, results_M, pe, times, experiment='Kuramoto', dim='0'):

   fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

   # Retrieve phase classification and temporal quantities
   phases = results_distrib[(c,k,m_bar)]         
   m = np.array(results_m[(c,k)])          
   M = np.array(results_M[(c,k)])          

   # Assign colors according to the detected phase
   colors = np.where(phases == -1, 'blue', np.where(phases == 0, 'yellow', 'red'))

   # =========================
   # 1) m(t)
   # =========================
   axes[0].plot(times, m, color='royalblue')
   axes[0].axhline(m_bar, linestyle='--', color='black')
   axes[0].set_title(fr"{experiment}: Temporal evolution of $m_{{c^*}}(t)$")

   # =========================
   # 2) M(t)
   # =========================
   axes[1].plot(times, M, color='crimson')
   axes[1].axhline(1-c, linestyle='--', color='black')
   axes[1].set_title(fr"{experiment}: Temporal evolution of $M_{{k^*}}(t)$")

   # =========================
   # 3) Entropy + phases
   # =========================
   axes[2].scatter(times, pe, c=colors, s=10)
   axes[2].plot(times, pe, alpha=0.3, color='gray')
   axes[2].set_title(fr"{experiment}: Temporal evolution of $PE(H{dim})(t)$ with phase classification")

   # Leyenda manual
   axes[2].scatter([], [], color='blue', label='Dispersion phase')
   axes[2].scatter([], [], color='red', label='Condensation phase')
   axes[2].scatter([], [], color='yellow', label='Undefined phase')
   axes[2].legend()

   plt.tight_layout()
   plt.show()