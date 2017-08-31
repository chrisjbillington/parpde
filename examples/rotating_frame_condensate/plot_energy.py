import h5py

with h5py.File('groundstate_8/0.h5', 'r') as f:
    sor_convergence = f['output_log']['convergence']

# with h5py.File('smoothing/0.h5', 'r') as g:
#     item_convergence = g['output_log']['convergence']

import matplotlib.pyplot as plt

plt.title('convergence')
plt.semilogy(sor_convergence[1:], label='SOR')
# plt.semilogy(item_convergence[1:], label='ITEM')
plt.legend()
plt.grid(True)



plt.figure()


sor_frac_change_per_step = 1 - sor_convergence[2:]/sor_convergence[1:-1]
# item_frac_change_per_step = 1 - item_convergence[2:]/item_convergence[1:-1]

plt.title('percent per step convergence rate')
plt.plot(100*sor_frac_change_per_step, label='SOR')
# plt.plot(100*item_frac_change_per_step, label='ITEM')
plt.axis(ymin=-10, ymax=10)
plt.legend()
plt.grid(True)
plt.show()
