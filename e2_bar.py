import matplotlib.pyplot as plt


# plt.bar([1,2,3], [142.86, 69.46, 46.56], label = "Independent computations")
# plt.bar([1,2,3], [0, 13.92, 11.77], bottom = [142.86, 69.46, 46.56], label = "Synchronization overhead")

plt.bar([1,2], [86.36,42.66], label = "Independent computations")
plt.bar([1,2], [0, 9.36], bottom = [86.36,42.66], label = "Synchronization overhead")

# plt.xticks([1,2,3], ('1 Node', '2 Nodes', '3 Nodes'))
plt.xticks([1,2], ('1 Node', '2 Nodes'))
plt.ylabel('Time (s)')
# plt.ylim([0,150])
plt.legend()
plt.show()