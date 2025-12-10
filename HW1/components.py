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

def draw_pca_first_n_VIF(df,vif_sorted,i):
    top_features = vif_sorted['feature'].iloc[-i:].tolist()
    print("Selected features for PCA:", top_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[top_features])
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Churn'] = df['Churn'].values
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'True.':'salmon', 'False.':'skyblue'}
    for label in pca_df['Churn'].unique():
        subset = pca_df[pca_df['Churn']==label]
        ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], 
                c=colors[label], label=label, alpha=0.6)

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title(f"3D PCA of Top {i} Features by VIF")
    ax.legend(title="Churn")
    plt.show()
    
    
    
