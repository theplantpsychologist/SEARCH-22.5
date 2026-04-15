import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import json

def generate_database_megaplot(db_path='database/storage/database_3.db', fig_dir="renders"):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # 1. FETCH DATA
    df = pd.read_sql_query("SELECT id, generation, has_children, layer_goodness, tree_efficiency, embedding FROM states", conn)
    df_branching = pd.read_sql_query("""
        SELECT p.generation, COUNT(c.id) as num_children
        FROM states p JOIN states c ON p.id = c.parent_id
        GROUP BY p.id
    """, conn)
    conn.close()

    # 2. CALCULATE METRICS
    def get_sparsity(buf):
        return np.count_nonzero(np.abs(np.frombuffer(buf, dtype=np.float32)) > 1e-6) if buf else 0
    df['complexity'] = df['embedding'].apply(get_sparsity)

    # 3. CALCULATE PREDICTION (G_{i+1} = n_{i+1} * G_i / p_i) with CI from children per parent
    counts = df.groupby('generation').agg(n_obs=('has_children', 'count'), p_obs=('has_children', 'sum')).sort_index()
    branching_stats = df_branching.groupby('generation')['num_children'].agg(['mean', 'std', 'count'])
    
    gens = counts.index.tolist()
    g_pred = np.zeros(len(gens)); g_sigma = np.zeros(len(gens)); g_ci = np.zeros(len(gens))
    g_pred[0] = counts.loc[gens[0], 'n_obs']
    cumulative_rel_var = 0; cumulative_rel_var_ci = 0

    for i in range(len(gens) - 1):
        n_next, p_curr = counts.loc[gens[i+1], 'n_obs'], counts.loc[gens[i], 'p_obs']
        if p_curr > 0 and n_next > 0:
            g_pred[i+1] = n_next * (g_pred[i] / p_curr)
            # Get children-per-parent stats for this generation
            if gens[i] in branching_stats.index:
                cpp_std = branching_stats.loc[gens[i], 'std']
                cpp_mean = branching_stats.loc[gens[i], 'mean']
                cpp_n = branching_stats.loc[gens[i], 'count']
                if cpp_mean > 0 and not np.isnan(cpp_std):
                    cumulative_rel_var += (cpp_std / cpp_mean) ** 2
                    cumulative_rel_var_ci += (1.96 * cpp_std / (np.sqrt(cpp_n) * cpp_mean)) ** 2
            g_sigma[i+1] = g_pred[i+1] * np.sqrt(cumulative_rel_var)
            g_ci[i+1] = g_pred[i+1] * np.sqrt(cumulative_rel_var_ci)
        else: 
            g_pred[i+1] = n_next

    # --- START PLOTTING ---
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    
    # [Row 1 & 2: Topological Metrics]
    metric_configs = [
        (df, 'layer_goodness', 'Mean Layer Goodness', axs[0, 0], '#2980b9'),
        (df, 'tree_efficiency', 'Mean Tree Efficiency', axs[0, 1], '#c0392b'),
        (df_branching, 'num_children', 'Avg Children per Parent', axs[1, 0], '#27ae60'),
        (df, 'complexity', 'Non-zero Eigenvalues', axs[1, 1], '#8e44ad')
    ]

    for data, col, title, ax, color in metric_configs:
        stats = data.groupby('generation')[col].agg(['mean', 'std', 'count']).dropna()
        x, m, s = stats.index, stats['mean'], stats['std']
        ci = 1.96 * (s / np.sqrt(stats['count']))
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.1, label='Std Dev')
        ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.3, label='95% CI')
        ax.plot(x, m, 'o-', color=color, linewidth=2, label='Mean')
        ax.set_title(title, fontweight='bold'); ax.grid(True, alpha=0.3); ax.legend(fontsize='small')

    # [Subplot 5: Current Expansion (Linear Bar)]
    ax_bar = axs[2, 0]
    p_vals = counts['p_obs'].values
    l_vals = counts['n_obs'].values - p_vals
    ax_bar.bar(gens, p_vals, color='#d63031', label='Expanded Parents')
    ax_bar.bar(gens, l_vals, bottom=p_vals, color='#ff7675', alpha=0.4, label='Childless States')
    ax_bar.set_title('Observed Generation Sizes (Linear)', fontweight='bold')
    ax_bar.set_ylabel('Number of States'); ax_bar.legend()

    # [Subplot 6: Predicted Potential (Log Line + Exp Fit)]
    ax_pred = axs[2, 1]
    # Filter for Gen > 2
    fit_mask = [i for i, g in enumerate(gens) if g > 2]
    fit_gens = np.array(gens)[fit_mask]
    fit_vals = g_pred[fit_mask]
    fit_sigma = g_sigma[fit_mask]
    fit_ci = g_ci[fit_mask]

    ax_pred.fill_between(fit_gens, fit_vals - fit_sigma, fit_vals + fit_sigma, color='#2d3436', alpha=0.1, label='Std Dev')
    ax_pred.fill_between(fit_gens, fit_vals - fit_ci, fit_vals + fit_ci, color='#2d3436', alpha=0.3, label='95% CI')
    ax_pred.plot(fit_gens, fit_vals, 'o-', color='#2d3436', linewidth=2, label='Predicted Total Size')

    # Exponential Fit: log(y) = k*x + b
    if len(fit_vals) > 1:
        slope, intercept, _, _, _ = linregress(fit_gens, np.log(fit_vals))
        fit_line = np.exp(intercept) * np.exp(slope * fit_gens)
        ax_pred.plot(fit_gens, fit_line, '--', color='#e67e22', label=f'Exp Fit (k={slope:.2f})')
        ax_pred.text(0.05, 0.05, f"Expansion Rate: {slope:.2f}\nDoubling every {0.693/slope:.1f} gens", 
                    transform=ax_pred.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    ax_pred.set_yscale('log')
    ax_pred.set_title('Predicted Search Frontier (Log Scale, Gen > 2)', fontweight='bold')
    ax_pred.set_ylabel('Predicted Potential States'); ax_pred.legend()

    plt.suptitle(f"Database Analytics Megaplot: {os.path.basename(db_path)}", fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    renders_dir = "renders"
    os.makedirs(renders_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(renders_dir) if f.endswith(".png")]
    file_count = len(existing_files)
    plt.savefig(os.path.join(renders_dir, f"db_megaplot_{file_count}.png"), dpi=300)
    print("Megaplot saved successfully.")


    def save_plot_data_to_json(db_path, df, df_branching, counts, branching_stats, gens, g_pred, g_sigma, g_ci, renders_dir="renders"):
        """
        For cross-platform visualization (manim)
        """
        
        # Calculate metrics
        def get_sparsity(buf):
            return np.count_nonzero(np.abs(np.frombuffer(buf, dtype=np.float32)) > 1e-6) if buf else 0
        df['complexity'] = df['embedding'].apply(get_sparsity)
        
        # Prepare data for JSON serialization
        plot_data = {
            "metadata": {
                "database_path": db_path,
                "database_name": os.path.basename(db_path),
                "database_size_bytes": os.path.getsize(db_path),
                "total_states": len(df),
                "total_generations": len(gens),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "generations": gens,
            "prediction": {
                "predicted": g_pred.tolist(),
                "sigma": g_sigma.tolist(),
                "ci_95": g_ci.tolist()
            },
            "metrics": {
                "layer_goodness": df.groupby('generation')['layer_goodness'].agg(['mean', 'std', 'count']).to_dict(),
                "tree_efficiency": df.groupby('generation')['tree_efficiency'].agg(['mean', 'std', 'count']).to_dict(),
                "complexity": df.groupby('generation')['complexity'].agg(['mean', 'std', 'count']).to_dict(),
                "children_per_parent": branching_stats[['mean', 'std', 'count']].to_dict()
            },
            "generation_sizes": {
                "n_obs": counts['n_obs'].to_dict(),
                "p_obs": counts['p_obs'].to_dict()
            }
        }
        
        json_path = os.path.join(renders_dir, f"db_megaplot_{len([f for f in os.listdir(renders_dir) if f.endswith('.png')])}_data.json")
        with open(json_path, 'w') as f:
            json.dump(plot_data, f, indent=2)
        print(f"Plot data saved to {json_path}")

    # Call this after line with plt.savefig():
    save_plot_data_to_json(db_path, df, df_branching, counts, branching_stats, gens, g_pred, g_sigma, g_ci, renders_dir)

if __name__ == "__main__":
    generate_database_megaplot()