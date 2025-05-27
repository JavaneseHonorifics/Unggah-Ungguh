import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from src.constants.mapping import map_labels, map_models

def visualize_radar_chart(input_dir: str, output_dir: str):
    df = pd.read_csv(f'{input_dir}/classification_report.csv')

    df['model'] = df['model'].apply(map_models)
    df['label'] = df['label'].apply(map_labels)

    df_grouped = df.groupby(['model', 'speaker', 'label'])[['precision', 'recall', 'f1-score', 'accuracy']].mean().reset_index()

    metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    models = df_grouped['model'].unique()
    labels = sorted(df_grouped['label'].unique())
    speakers = sorted(df_grouped['speaker'].unique())

    fig = plt.figure(figsize=(40, 18))
    gs = fig.add_gridspec(2, 5, width_ratios=[0.1, 1, 1, 1, 1], height_ratios=[1, 1], hspace=0.1, wspace=0.5)

    color_palette = [
        '#360A4F', '#4D1772', '#6D2193',
        '#A0388D', '#D74D7E', '#E0818F', '#EFB69D'
    ]

    legend_handles = []
    legend_labels = []

    for idx, speaker in enumerate(speakers):
        ax_label = fig.add_subplot(gs[idx, 0])
        ax_label.text(1.9, 0.5, speaker.replace(' Utterance', ''),
                      rotation=0, ha='center', va='center',
                      fontsize=24, fontweight='bold')
        ax_label.axis('off')

    for speaker_idx, speaker in enumerate(speakers):
        for label_idx, label in enumerate(labels):
            ax = fig.add_subplot(gs[speaker_idx, label_idx + 1], projection='polar')
            filtered_data = df_grouped[(df_grouped['label'] == label) & (df_grouped['speaker'] == speaker)]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
            ax.set_xticks(angles)

            for idx, label_metric in enumerate(metrics):
                angle = angles[idx]
                if angle == 0:
                    ha, va = 'center', 'bottom'
                elif 0 < angle < np.pi:
                    ha, va = 'left', 'center'
                elif angle == np.pi:
                    ha, va = 'center', 'top'
                else:
                    ha, va = 'right', 'center'
                text = ax.text(angle, 1.02, label_metric, ha=ha, va=va, size=17)

            ax.set_xticklabels([])

            for model_idx, model in enumerate(models):
                model_data = filtered_data[filtered_data['model'] == model]
                if not model_data.empty:
                    values = model_data[metrics].values[0]
                    plot_values = list(values) + [values[0]]
                    plot_angles = np.concatenate((angles, [angles[0]]))

                    ax.plot(plot_angles, plot_values, color=color_palette[model_idx], linewidth=2.5)

                    if speaker_idx == 0 and label_idx == 0:
                        rect = Rectangle((0, 0), 1, 1, facecolor=color_palette[model_idx], alpha=1)
                        legend_handles.append(rect)
                        legend_labels.append(model)

            ax.set_title(label, size=24, fontweight='semibold', pad=35)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=1)

    fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.1), ncol=4, fontsize=32)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{output_dir}/model_performance_radar.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()