import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 40})
plt.rcParams.update({"font.weight": 'bold'})
plt.rcParams.update({"font.family": 'Palatino Linotype'})


def ts_compare_baselines():
    plt.figure(figsize=(6.6, 6))
    plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)

    # optimum_cpu = [40.5, 42, 47]
    # algorithm_cpu = [42.5, 46.4, 52.5]
    # rule_based_cpu = [0, 0, 0]

    bar_width = 0.25
    optimum = [1, 1.0, 1.0]
    algorithm = [1.020531401, 1.044170414, 1.055851064]
    rule_based = [1.096618357, 1.157175399, 1.234]
    algorithm_std_cpu = [0.013502826, 0.014057585, 0.027510852]

    r1 = np.arange(len(optimum))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Make the plot
    plt.bar(r1, optimum, color='#005CAB', width=bar_width, edgecolor='black', label='OPTM', zorder=5)
    plt.bar(r2, algorithm, color='#E31B23', width=bar_width, edgecolor='black', label='PEMA', zorder=5)
    plt.bar(r3, rule_based, color='#ffbf00', width=bar_width, edgecolor='black', label='RULE', zorder=5)

    # Add xticks on the middle of the group bars
    plt.ylabel('Normalized CPU', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(optimum))], ['125', '225', '325'])
    plt.yticks(np.arange(0, 1.6, 0.5))
    plt.ylim(0, 1.7)
    # Create legend & Show graphic
    plt.xlabel("Workloads (RPS)", fontweight='bold')
    plt.legend(loc='upper left', ncol=2, fontsize=32, frameon=False, borderaxespad=0.02, handlelength=0.7,
               handletextpad=0.2, labelspacing=0.2, columnspacing=0.6, bbox_to_anchor=(0.008, 1.01))
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig("ts_performance_comparisons.pdf", bbox_inches='tight')
    plt.show()