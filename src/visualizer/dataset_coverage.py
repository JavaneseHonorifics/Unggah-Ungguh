import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from src.constants.results import ngoko, ngoko_alus, krama, krama_alus
from src.constants.mapping import JAVANESE_CORPUS

def _calculate_entropy(percentages):
    probabilities = [p/100 for p in percentages if p > 0]
    entropy = -sum([p * math.log2(p) for p in probabilities])
    return entropy

def visualize_dataset_distribution(output_dir: str):
    entropies = []
    for i in range(len(JAVANESE_CORPUS)):
        dataset_distribution = [ngoko[i], ngoko_alus[i], krama[i], krama_alus[i]]
        entropies.append(_calculate_entropy(dataset_distribution))

    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    fig.patch.set_facecolor('white')

    colors = ['#28104E', '#6237A0', '#9754CB', '#DEACF5']

    data_matrix = np.array([ngoko, ngoko_alus, krama, krama_alus])
    unggah_ungguh_index = JAVANESE_CORPUS.index("Unggah-Ungguh")

    annot_data = np.array([[f'$\mathbf{{{v:.1f}}}$' if j == unggah_ungguh_index else f'{v:.1f}' 
                            for j, v in enumerate(row)] for row in data_matrix])

    sns.heatmap(data_matrix, ax=ax1, cmap='viridis', 
                xticklabels=JAVANESE_CORPUS, 
                yticklabels=['Ngoko', 'Ngoko Alus', 'Krama', 'Krama Alus'],
                annot=annot_data, fmt='', cbar_kws={'label': 'Percentage (%)'})

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20, ha='right')

    bars = ax3.bar(np.arange(len(JAVANESE_CORPUS)), entropies, color='#28104E', alpha=1)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        if i == unggah_ungguh_index:
            bar.set_hatch('///')
            bar.set_edgecolor('white')
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', fontweight='bold', ha='center', va='bottom')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}', ha='center', va='bottom')

    ax3.set_ylabel('Entropy Value', fontsize=12, fontweight='bold')
    ax3.set_xticks(np.arange(len(JAVANESE_CORPUS)))
    ax3.set_xticklabels([r'$\bf{Unggah-Ungguh}$' if d == 'Unggah-Ungguh' else d for d in JAVANESE_CORPUS], rotation=20, ha='right')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_path = f"{output_dir}/dataset_coverage.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()