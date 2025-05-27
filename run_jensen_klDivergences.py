from src.compute_stats import compute_divergence_json

if __name__ == "__main__":
    compute_divergence_json(
        ds="JavaneseHonorifics/Unggah-Ungguh",
        subset="translation",
        split="train",
        output_dir="./results/jensen-klDivergence.json"
    )