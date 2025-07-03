import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import gdown

@st.cache_data
def load_data_from_gdrive(file_id):
    """
    Downloads file from Google Drive using gdown and loads into pandas DataFrame.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "temp.csv"
    gdown.download(url, output, quiet=True)
    return pd.read_csv(output)

def interactive_protein_heatmap_with_sites(protein_df, threshold=0.5):
    # ... your existing plotting function unchanged ...
    protein_df = protein_df.copy()
    protein_df['predicted_binary'] = (protein_df['predicted_score_bad'] > threshold).astype(int)

    residues = protein_df['AA'].tolist()
    positions = list(range(len(protein_df)))

    ss_map = {'-': 'gray', 'E': 'blue', 'H': 'red', 'S': 'orange', 'P': 'purple', 'T': 'green', 'B': 'darkblue', 'I': 'darkgreen', 'G': 'orange'}

    top_heatmap_z = np.array([
        protein_df['known_peptide'].values,
        protein_df['predicted_binary'].values,
        protein_df['predicted_score_bad'].values
    ])

    bottom_heatmap_z = np.array([
        protein_df['conservation_norm'].fillna(0).values,
        protein_df['pathogenicity'].fillna(0).values
    ])

    def normalize(arr):
        arr = np.array(arr)
        arr = np.nan_to_num(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

    bottom_heatmap_z[0] = normalize(bottom_heatmap_z[0])
    bottom_heatmap_z[1] = normalize(bottom_heatmap_z[1])

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.05, 0.3, 0.15, 0.03, 0.03],
        subplot_titles=[
            'Residues',
            ' ',
            'Conservation | Pathogenicity',
            'Secondary Structure (SS)',
            'Sites'
        ]
    )

    fig.add_trace(go.Heatmap(
        z=[[0] * len(positions)],
        x=positions,
        y=['Residue'],
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hoverinfo='skip',
        text=[residues],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "black"},
        xgap=0,
        ygap=0
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=top_heatmap_z,
        x=positions,
        y=['True Label', 'Binary Prediction', 'Predicted Score'],
        colorscale='Viridis',
        zmin=0, zmax=1,
        showscale=False,
        hoverinfo='x+y+z',
        xgap=0, ygap=0,
        zsmooth='best',
        showlegend=False
    ), row=2, col=1)

    sep_lines = []
    for y in [0.5, 1.5]:
        sep_lines.append(dict(
            type="line",
            x0=-0.5,
            x1=len(positions) - 0.5,
            y0=y,
            y1=y,
            xref='x',
            yref='y2',
            line=dict(color='white', width=2),
            layer="above"
        ))

    fig.add_trace(go.Heatmap(
        z=bottom_heatmap_z,
        x=positions,
        y=['Conservation', 'Pathogenicity'],
        colorscale='Blues',
        zmin=0, zmax=1,
        showscale=False,
        hoverinfo='x+y+z',
        xgap=0, ygap=0,
        zsmooth='best'
    ), row=3, col=1)

    sep_lines.append(dict(
        type="line",
        x0=-0.5,
        x1=len(positions) - 0.5,
        y0=0.5,
        y1=0.5,
        xref='x',
        yref='y3',
        line=dict(color='white', width=2),
        layer="above"
    ))

    ss_map_full = {
        "H": "Alpha helix (4-12)",
        "B": "Isolated beta-bridge residue",
        "E": "Strand",
        "G": "3-10 helix",
        "I": "Pi helix",
        "T": "Turn",
        "S": "Bend",
        "P": "Kappa helix",
        "-": "Unknown"
    }

    unique_ss_codes = list(ss_map_full.keys())
    ss_numeric = [unique_ss_codes.index(ss) if ss in unique_ss_codes else unique_ss_codes.index('-') for ss in protein_df['SS'].fillna('-')]

    ss_heatmap_colors = [ss_map.get(code, 'gray') for code in unique_ss_codes]
    ss_colorscale = [[i / max(len(ss_heatmap_colors) - 1, 1), color] for i, color in enumerate(ss_heatmap_colors)]

    hover_ss = [
        ss_map_full.get(ss, 'Unknown') for ss in protein_df['SS'].fillna('-')
    ]

    fig.add_trace(go.Heatmap(
        z=[ss_numeric],
        x=positions,
        y=['SS'],
        colorscale=ss_colorscale,
        showscale=False,
        hoverinfo='text',
        text=[hover_ss],
        xgap=0,
        ygap=0
    ), row=4, col=1)

    for r in range(1, 5):
        fig.update_xaxes(
            tickmode='array',
            tickvals=positions,
            ticktext=[''] * len(positions),
            showgrid=False,
            row=r, col=1
        )

    site_columns = [col for col in protein_df.columns if col.startswith('sites_')]
    protein_df['sites'] = protein_df[site_columns].idxmax(axis=1).str.replace('sites_', '')
    protein_df.loc[protein_df[site_columns].sum(axis=1) == 0, 'sites'] = 'None'

    site_types = sorted(protein_df['sites'].fillna('None').unique())
    site_map = {site: i for i, site in enumerate(site_types)}
    site_colors_list = ['pink', 'cyan', 'yellow', 'brown', 'lightgreen', 'orange', 'purple', 'gray']
    site_colorscale = [[i / max(len(site_map) - 1, 1), site_colors_list[i % len(site_colors_list)]] for i in range(len(site_map))]

    site_numeric = [site_map.get(site, site_map.get('None', 0)) for site in protein_df['sites'].fillna('None')]
    hover_sites = [f"Site: {site}" for site in protein_df['sites'].fillna('None')]

    fig.add_trace(go.Heatmap(
        z=[site_numeric],
        x=positions,
        y=['sites'],
        colorscale=site_colorscale,
        showscale=False,
        hoverinfo='text',
        text=[hover_sites],
        xgap=0,
        ygap=0
    ), row=5, col=1)

    for r in range(1, 6):
        fig.update_xaxes(
            tickmode='array',
            tickvals=positions,
            ticktext=[''] * len(residues),
            showgrid=False,
            row=r, col=1
        )

    for site_type, color_index in zip(site_types, range(len(site_types))):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=site_colors_list[color_index % len(site_colors_list)]),
            name=f'Site: {site_type}',
            showlegend=True
        ))

    fig.update_layout(
        shapes=sep_lines,
        height=850,
        width=max(900, len(protein_df) * 18),
        title_text=f"Protein Visualization — Gene: {protein_df['gene'].iloc[0]} | Accession: {protein_df['accession'].iloc[0]}",
        hovermode='x unified',
        font=dict(size=12, family='Arial, serif'),
        margin=dict(t=140, b=80),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12),
            traceorder="normal"
        )
    )

    return fig

# --- Streamlit UI ---

st.title("Protein Interactive Visualization")

# Use just the file ID here:
FILE_ID = "1DzSyx8MjTP2LDWK6a3i2hzvxSrZIe8eT"

df = load_data_from_gdrive(FILE_ID)

# Find genes with known peptides
genes_with_peptides = df.groupby('gene')['known_peptide'].max()
genes_with_peptides = genes_with_peptides[genes_with_peptides == 1].index.tolist()

selected_gene = st.selectbox("Select a gene", options=genes_with_peptides)

protein_df = df[df['gene'] == selected_gene].copy()
protein_df.reset_index(drop=True, inplace=True)

threshold = st.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

fig = interactive_protein_heatmap_with_sites(protein_df, threshold=threshold)

st.plotly_chart(fig, use_container_width=True)
