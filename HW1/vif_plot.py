import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


def interactive_embedding_explorer(df):
    try:
        import umap.umap_ as umap
        UMAP_AVAILABLE = True
    except Exception:
        UMAP_AVAILABLE = False

    df_work = df.copy()
    df_work['Churn_str'] = df_work['Churn'].astype(str).str.strip()
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')

    if len(numeric_cols) < 2:
        raise ValueError("Not enough numeric features for VIF / embeddings. Numeric columns: " + str(numeric_cols))

    X_vif = df_work[numeric_cols].fillna(0).astype(float)
    vif_vals = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    vif_df = pd.DataFrame({
        'feature': X_vif.columns,
        'VIF': vif_vals
    })
    vif_df['VIF_display'] = vif_df['VIF'].map(lambda x: f"{x:.2f}")

    vif_sorted = vif_df.sort_values(by='VIF', ascending=False).reset_index(drop=True)
    print("VIF (sorted desc):")
    display(vif_sorted[['feature','VIF_display']])

    # Widgets
    method_options = ['PCA', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else []) + ['None']
    method_dd = widgets.Dropdown(options=method_options, value='PCA', description='Method:')
    dim_dd = widgets.Dropdown(options=[2,3], value=2, description='Dims:')
    perplexity_slider = widgets.IntSlider(value=30, min=5, max=100, step=1, description='t-SNE perplexity:')
    n_neighbors_slider = widgets.IntSlider(value=15, min=2, max=100, step=1, description='UMAP n_neighbors:')

    # Create checkboxes storing real feature names (sorted by VIF desc)
    feature_checkboxes = []
    for idx, row in vif_sorted.iterrows():
        feat = row['feature']
        vif_str = row['VIF_display']
        cb = widgets.Checkbox(value=(idx < 6), description=f"{feat} (VIF={vif_str})", indent=False)
        cb.feature_name = feat
        feature_checkboxes.append(cb)

    checkbox_box = widgets.VBox(feature_checkboxes, layout=widgets.Layout(overflow='auto', max_height='360px'))

    # Buttons and output area
    select_all_btn = widgets.Button(description="Select all", layout=widgets.Layout(width='110px'))
    deselect_all_btn = widgets.Button(description="Deselect all", layout=widgets.Layout(width='110px'))
    apply_btn = widgets.Button(description='Update Plot', button_style='primary', layout=widgets.Layout(width='140px'))
    out_box = widgets.Output()

    # Helpers
    def get_selected_features():
        selected = [cb.feature_name for cb in feature_checkboxes if cb.value]
        if len(selected) == 0:
            selected = vif_sorted['feature'].iloc[:6].tolist()
        return selected

    def compute_embedding(features, method='PCA', dims=2, perplexity=30, n_neighbors=15, random_state=42):
        X = df_work[features].fillna(0).astype(float).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        if method == 'PCA':
            pca = PCA(n_components=dims, random_state=random_state)
            Z = pca.fit_transform(Xs)
        elif method == 't-SNE':
            ts = TSNE(n_components=dims, random_state=random_state, init='pca', perplexity=perplexity)
            Z = ts.fit_transform(Xs)
        elif method == 'UMAP' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=dims, random_state=random_state, n_neighbors=n_neighbors)
            Z = reducer.fit_transform(Xs)
        elif method == 'None':
            needed = dims
            Z = Xs[:, :needed] if Xs.shape[1] >= needed else np.hstack([Xs, np.zeros((Xs.shape[0], needed - Xs.shape[1]))])
        else:
            raise ValueError("Selected method not available.")
        col_names = [f"Dim{i+1}" for i in range(dims)]
        out = pd.DataFrame(Z, columns=col_names)
        out['Churn_str'] = df_work['Churn_str'].values
        return out

    def plot_embedding(emb_df, dims=2, title="Embedding"):
        # prepare color map while tolerating different churn label variants
        mapping = {'True.':'salmon', 'False.':'skyblue', 'True':'salmon', 'False':'skyblue', 'true':'salmon', 'false':'skyblue'}
        unique_vals = sorted(emb_df['Churn_str'].unique())
        color_map = {val: mapping.get(val, None) for val in unique_vals}
        # if none of mapping matched, Plotly will pick automatic colors
        if dims == 2:
            fig = px.scatter(
                emb_df, x='Dim1', y='Dim2', color='Churn_str',
                title=title, opacity=0.7,
                color_discrete_map=color_map if any(color_map.values()) else None
            )
        else:
            fig = px.scatter_3d(
                emb_df, x='Dim1', y='Dim2', z='Dim3', color='Churn_str',
                title=title, opacity=0.7,
                color_discrete_map=color_map if any(color_map.values()) else None
            )
        fig.update_layout(height=700, legend_title_text='Churn')
        fig.show()

    # Callbacks
    def on_select_all(b):
        for cb in feature_checkboxes:
            cb.value = True

    def on_deselect_all(b):
        for cb in feature_checkboxes:
            cb.value = False

    def on_apply(b):
        with out_box:
            clear_output(wait=True)
            features = get_selected_features()
            missing = [f for f in features if f not in df_work.columns]
            if missing:
                print("These selected features are not in df.columns:", missing)
                print("Available columns:", df_work.columns.tolist())
                return
            if len(features) < 2:
                print("Select at least 2 features.")
                return
            method = method_dd.value
            dims = dim_dd.value
            perplexity = perplexity_slider.value
            n_neighbors = n_neighbors_slider.value
            print("Using features:", features)
            try:
                emb = compute_embedding(features, method=method, dims=dims, perplexity=perplexity, n_neighbors=n_neighbors)
            except Exception as e:
                print("Error during embedding:", e)
                return
            title = f"{method} ({dims}D) — features: {', '.join(features)}"
            plot_embedding(emb, dims=dims, title=title)

    select_all_btn.on_click(on_select_all)
    deselect_all_btn.on_click(on_deselect_all)
    apply_btn.on_click(on_apply)

    # Layout display
    controls_left = widgets.VBox([
        method_dd,
        dim_dd,
        widgets.HBox([perplexity_slider, n_neighbors_slider]),
        widgets.HBox([select_all_btn, deselect_all_btn, apply_btn])
    ])
    ui = widgets.HBox([controls_left, checkbox_box], layout=widgets.Layout(align_items='flex-start', gap='30px'))
    display(ui, out_box)

    # initial run
    on_apply(None)

    # return references in case user wants to access them
    return {
        'ui': ui,
        'out_box': out_box,
        'feature_checkboxes': feature_checkboxes,
        'method_dd': method_dd,
        'dim_dd': dim_dd,
        'apply_btn': apply_btn
    }

