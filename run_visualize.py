from src.visualizer.dataset_coverage import visualize_dataset_distribution
from src.visualizer.dataset_radar import visualize_radar_chart

if __name__ == "__main__":
    visualize_dataset_distribution(
        output_dir="./results/visualization"
    )
    visualize_radar_chart(
        input_dir="./results/task4",
        output_dir="./results/visualization"
    )
