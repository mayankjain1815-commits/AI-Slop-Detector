import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt


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


def get_results_by_domain(df: pd.DataFrame, agg="predictions_take_max") -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    domains = set(df["domain"])
    results = {}

    for domain in domains:
        df_filter = df[df["domain"] == domain]
        y_true = get_y_true(df_filter["model"])
        y_score = df_filter[agg]

        results[domain] = roc_curve(y_true, y_score)

    return results


def plot_tpr_bars(
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


def plot_roc(
    domain_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    fig_path: str,
) -> None:
    domains = ['reviews', 'books', 'wiki', 'reddit', 'news', 'abstracts', 'poetry', 'recipes']

    fig, ax = plt.subplots(figsize=(5, 4))

    for idx, domain in enumerate(domains):
        fpr, tpr, _ = domain_results[domain]
        ax.plot(100 * fpr, 100 * tpr, label=domain.capitalize(), alpha=(1.0 if idx < 4 else 0.5))

    ax.plot([0, 100], [0, 100], 'k--')

    ax.set_xlabel("FPR (%)")
    ax.set_ylabel("TPR (%)")
    ax.legend(ncol=2)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    

def main():
    df = load_data("./analysis/predictions_10000.csv")

    domain_results = get_results_by_domain(df)
    
    plot_tpr_bars(domain_results, './analysis/tpr_bars.pdf')
    plot_roc(domain_results, './analysis/roc.pdf')


if __name__ == '__main__':
    main()