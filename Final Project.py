#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import signal
import matplotlib.pyplot as plt
import control as ctl
import numpy as np
from scipy import interpolate

#information of the plant
num_p = [1]
den_p = [1, 11.1, 11.1, 1]
lti = signal.lti(num_p, den_p)
tp, yp = signal.step(lti)
sysc = ctl.TransferFunction(num_p, den_p)
print("Gp(s) = \n", sysc)

#plot step response of the plant
plt.plot(tp, yp)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Gp(s)')
plt.grid()


# In[2]:


#the parameter is designed
omega = 3
eta = 0.5
alpha = 5*omega
beta = 6*omega
gamma = 7*omega

#calculation of every parameters
qus = np.poly1d([1, 2*eta*omega, omega**2])
poles = np.poly1d([1, (alpha + beta + gamma), (alpha*beta + beta*gamma + gamma*alpha), alpha*beta*gamma])
Den = qus*poles
B1_B2 = Den[4] - 11.1
B1B2 = Den[3] - 11.1 - B1_B2*11.1
k = Den[2] - 1 - B1B2 - B1_B2*11.1
A1_A2 = (Den[1] - B1_B2 - B1B2*11.1)/k
A1A2 = (Den[0] - B1B2)/k

#show parameters information
print("B1+B2 = ", B1_B2)
print("B1*B2 = ", B1B2)
print("k = ", k)
print("A1+A2 = ", A1_A2)
print("A1A2 = ", A1A2)

#transfer function
num_5 = [k, k*A1_A2, k*A1A2]
den_5 = [Den[5], Den[4], Den[3], Den[2], Den[1], Den[0]]

#show information of the transfer function
lti = signal.lti(num_5, den_5)
t, y = signal.step(lti)
sysc = ctl.TransferFunction(num_5, den_5)
print("\nTF(s) =", sysc)
Gcs = ctl.TransferFunction([k, k*(A1_A2), k*A1A2], [1, B1_B2, B1B2])
print("Gc(s) = ", Gcs)

#plot step response
plt.figure(figsize=(8,5))
plt.plot(t, y)
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Amplitude', size = 16)
plt.title('Step Response', size = 16)
plt.axhline(0.95, ls='--', c='red', label = "5%")
plt.axhline(1.05, ls='--', c='red')
plt.legend(loc = 'lower right', prop={'size': 16})
plt.grid()


# In[3]:


#show information
print("TF =", sysc)
sysd_01 = sysc.sample(0.1, method='matched')
print("TF(z) =", sysd_01)
Gcsd = Gcs.sample(0.1, method='matched')
print("Gc(z) = ", Gcsd)

#picture of T = 0.1 TF(z)
sysd_01 = signal.dlti(sysd_01.num[0][0], sysd_01.den[0][0], dt=0.1)
plt.figure(figsize=(8,5))
td_01, yd_01 = signal.dstep(sysd_01, n=50)
plt.step(td_01, np.squeeze(yd_01)/np.squeeze(yd_01)[-1])
plt.grid()
plt.xlabel('n [samples]', size = 16)
plt.ylabel('Amplitude', size = 16)
plt.title('discrete T=0.1', size = 16)

# for T = 0.05
sysd_005 = sysc.sample(0.05, method='matched')
sysd_005 = signal.dlti(sysd_005.num[0][0], sysd_005.den[0][0], dt=0.05)
td_005, yd_005 = signal.dstep(sysd_005, n=100)


# In[4]:


#discrete to continuous
cont_01 = interpolate.interp1d(td_01, np.squeeze(yd_01)/np.squeeze(yd_01)[-1], kind="linear")
cont_005 = interpolate.interp1d(td_005, np.squeeze(yd_005)/np.squeeze(yd_005)[-1], kind="linear")

tcont = np.arange(0, 5, 0.1)
ycont_01 = cont_01(tcont)
ycont_005 = cont_005(tcont)

plt.figure(figsize=(8,5))
plt.plot(tcont, ycont_01, label = "T=0.1")
plt.plot(tcont, ycont_005, label = "T=0.05")
plt.plot(t, y, label = "continuous")
plt.grid()
plt.xlabel('time[s]', size = 16)
plt.ylabel('Amplitude', size = 16)
plt.axhline(0.95, ls='--', c='red', label = "5%")
plt.axhline(1.05, ls='--', c='red')
plt.legend(prop={'size': 16})
plt.title('Continuous & Discrete Step Response', size = 16)
plt.show()


# In[5]:


#state feedback computation
open_loop_den_p1 = np.poly1d([1, B1_B2, B1B2])
open_loop_den_p2 = np.poly1d([1, 11.1, 11.1, 1])
print("open loop denominator = \n", open_loop_den_p1*open_loop_den_p2)

print("\nclose loop denominator = \n", Den)

feedback_gain = Den - open_loop_den_p1*open_loop_den_p2
print("\nk value = \n", [feedback_gain[0], feedback_gain[1], feedback_gain[2], feedback_gain[3], feedback_gain[4]])


# In[6]:


#test of lead compensater form
b = [24.9, 23.59]
k = [829.61, 844.1]
a = [2.899, 2.851]

num_1 = [k[0], k[0]*a[0]]
den_1 = [1, 36, 378, 1107, 2430]
lti1 = signal.lti(num_1, den_1)
t1, y1 = signal.step(lti1)
sysc = ctl.TransferFunction(num_1, den_1)
print("Gp1(s) = \n", sysc)

num_2 = [k[1], k[1]*a[1]]
den_2 = [1, 36, 378, 1107, 2430]
lti2 = signal.lti(num_2, den_2)
t2, y2 = signal.step(lti2)
sysc = ctl.TransferFunction(num_2, den_2)
print("Gp2(s) = \n", sysc)

plt.figure(figsize=(8,5))
plt.plot(t1, y1, label = "1")
plt.plot(t2, y2, label = "2")
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Amplitude', size = 16)
plt.title('Step Response with failed Design', size = 16)
plt.legend(prop={'size': 16})
plt.grid()


# In[ ]:




