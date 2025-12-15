import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

DOMAINS = ['reviews', 'books', 'wiki', 'reddit', 'news', 'abstracts', 'poetry', 'recipes']
MODEL_FAMILIES = ["llama", "gpt", "cohere",  "mistral", "mpt"]
MODEL_NAMES = ["Llama", "GPT", "Cohere",  "Mistral", "MPT"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # This aggregator is worse than random guessing...
    df["predictions_all_or_nothing"] = 1 - df["predictions_all_or_nothing"]
    
    return df


def get_y_true(models: pd.Series) -> pd.Series:
    return (models != "human").astype(int)


def get_tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    
    return float(tpr[idx[-1]])


def get_results_by_domain(
    df: pd.DataFrame,
    agg="predictions_take_max"
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    domains = set(df["domain"])
    results = {}

    for domain in domains:
        df_filter = df[df["domain"] == domain]
        y_true = get_y_true(df_filter["model"])
        y_score = df_filter[agg]

        results[domain] = roc_curve(y_true, y_score)

    return results


def plot_domain_tpr_bars(
    domain_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    fig_path: str,
) -> None:
    # Calculate
    tpr_at_5 = {}
    tpr_at_1 = {}
    for domain, (fpr, tpr, _) in domain_results.items():
        tpr_at_5[domain] = get_tpr_at_fpr(fpr, tpr, 0.05)
        tpr_at_1[domain] = get_tpr_at_fpr(fpr, tpr, 0.01)

    # Plot
    bar_width = 0.33
    offset = bar_width/2
    domains = sorted(domain_results.keys(), key=lambda domain: tpr_at_5[domain], reverse=True)
    tpr_at_5_list = [100 * tpr_at_5[domain] for domain in domains]
    tpr_at_1_list = [100 * tpr_at_1[domain] for domain in domains]
    bar_x = np.arange(len(domains))

    fig, ax = plt.subplots(figsize=(7, 3))

    ax.bar(bar_x - offset, tpr_at_5_list, bar_width, label="FPR = 5%")
    ax.bar(bar_x + offset, tpr_at_1_list, bar_width, label="FPR = 1%")

    ax.set_xticks(bar_x)
    ax.set_xticklabels([domain.capitalize() for domain in domains])
    
    ax.set_xlabel("Domain")
    ax.set_ylabel("TPR (%)")
    ax.legend()

    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')


def plot_subresult_roc(
    subresults: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    keys: list[str],
    fig_path: str,
    key_names: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    for idx, key in enumerate(keys):
        fpr, tpr, _ = subresults[key]

        label = key_names[idx] if key_names else key.capitalize()
        alpha = 1.0 if idx < 4 or len(keys) <= 5 else 0.5

        ax.plot(100 * fpr, 100 * tpr, label=label, alpha=alpha)

    ax.plot([0, 100], [0, 100], 'k--')

    ax.set_xlabel("FPR (%)")
    ax.set_ylabel("TPR (%)")
    ax.legend(ncol=2)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')


def get_results_by_model_families(
    df: pd.DataFrame,
    agg="predictions_take_max"
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    results = {}

    for family in MODEL_FAMILIES:
        df_filter = df[df["model"].str.contains(f"{family}|human")]

        y_true = get_y_true(df_filter["model"])
        y_score = df_filter[agg]

        results[family] = roc_curve(y_true, y_score)

    return results


def main():
    df = load_data("./analysis/predictions_10000.csv")

    # Results by model
    model_results = get_results_by_model_families(df)
    plot_subresult_roc(model_results, MODEL_FAMILIES, './analysis/roc-model.pdf', MODEL_NAMES)

    for domain in ["wiki", "reviews", "books"]:
        df_filter = df[df["domain"] == domain]
        filter_model_results = get_results_by_model_families(df_filter)
        plot_subresult_roc(filter_model_results, MODEL_FAMILIES, f'./analysis/roc-model-{domain}.pdf', MODEL_NAMES)

    # Results by domain
    domain_results = get_results_by_domain(df)
    
    plot_domain_tpr_bars(domain_results, './analysis/tpr_bars.pdf')
    plot_subresult_roc(domain_results, DOMAINS, './analysis/roc-domain.pdf')


if __name__ == '__main__':
    main()