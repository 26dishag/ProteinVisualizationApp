import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import zipfile
import io
from scipy.ndimage import gaussian_filter1d

@st.cache_data
def load_data_from_zip(uploaded_zip):
    """
    Loads CSV data from an uploaded ZIP file in Streamlit.
    """
    with zipfile.ZipFile(uploaded_zip) as z:
        # Assumes the first file inside the zip is the CSV
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as csvfile:
            df = pd.read_csv(csvfile)
    return df

def interactive_protein_heatmap_with_sites(protein_df, threshold=0.5):
    protein_df = protein_df.copy()
    protein_df['predicted_binary'] = (protein_df['predicted_score_bad'] > threshold).astype(int)

    residues = protein_df['AA'].tolist()
    positions = list(range(len(protein_df)))

    # Residues: bold + red if part of known peptide, else normal black
    residues_styled = [
        f"<b><span style='color:red'>{res}</span></b>" if kp else res
        for res, kp in zip(residues, protein_df['known_peptide'].astype(bool))
    ]

    def normalize(arr):
        arr = np.nan_to_num(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

    # Data rows (order below residues)
    data_rows = {
        'Predicted Score': normalize(protein_df['predicted_score_bad'].values),
        'Binary Prediction': normalize(protein_df['predicted_binary'].values),
        'Known Peptide': normalize(protein_df['known_peptide'].values),
        'Conservation': normalize(protein_df['conservation_norm'].fillna(0).values),
        'Pathogenicity': normalize(protein_df['pathogenicity'].fillna(0).values),
        'relASA': normalize(protein_df['relASA'].fillna(0).values),
    }

    sigma_horizontal = 0.2
    for key in ['Predicted Score', 'Binary Prediction', 'Conservation', 'Pathogenicity', 'relASA']:
        data_rows[key] = gaussian_filter1d(data_rows[key], sigma=sigma_horizontal)

    # Pastel sequential colorscale (custom) for Predicted Score etc.
    pastel_seq = [
        [0.0, '#fce4ec'],  # very light pink
        [0.2, '#f8bbd0'],
        [0.4, '#f48fb1'],
        [0.6, '#f06292'],
        [0.8, '#ec407a'],
        [1.0, '#e91e63']   # deeper pink
    ]

    # Secondary structure colors - complementary pastel palette
    ss_map_full = {
        "H": "Alpha helix (4-12)", "B": "Isolated beta-bridge residue", "E": "Strand",
        "G": "3-10 helix", "I": "Pi helix", "T": "Turn", "S": "Bend", "P": "Kappa helix", "-": "Unknown"
    }
    ss_codes = list(ss_map_full.keys())
    ss_numeric = [ss_codes.index(ss) if ss in ss_codes else ss_codes.index('-') for ss in protein_df['SS'].fillna('-')]
    ss_colors = ['#a8dadc', '#457b9d', '#1d3557', '#f1faee', '#e63946', '#ffb703', '#fb8500', '#b56576', '#adb5bd']
    ss_colorscale = [[i/(len(ss_colors)-1), c] for i,c in enumerate(ss_colors)]

    # Sites colors - complementary pastel palette
    site_cols = [col for col in protein_df.columns if col.startswith('sites_')]
    protein_df['sites'] = protein_df[site_cols].idxmax(axis=1).str.replace('sites_', '')
    protein_df.loc[protein_df[site_cols].sum(axis=1) == 0, 'sites'] = 'None'
    site_types = sorted(protein_df['sites'].unique())
    site_map = {s:i for i,s in enumerate(site_types)}
    site_colors_list = ['#ffadad', '#ffd6a5', '#fdffb6', '#caffbf', '#9bf6ff', '#a0c4ff', '#bdb2ff', '#ffc6ff']
    site_colorscale = [[i/(len(site_types)-1), site_colors_list[i % len(site_colors_list)]] for i in range(len(site_types))]
    site_numeric = [site_map[s] for s in protein_df['sites']]

    # Define rows without extra legend row
    all_rows = ['Residues'] + list(data_rows.keys()) + ['Secondary Structure', 'Sites']
    n_heatmap_rows = len(all_rows)
    row_height = 0.05
    row_heights = [row_height] * n_heatmap_rows

    fig = make_subplots(
        rows=n_heatmap_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
        row_heights=row_heights,
        subplot_titles=[None]*n_heatmap_rows
    )

    # Residues row (top)
    fig.add_trace(go.Heatmap(
        z=[[0]*len(positions)],
        x=positions,
        y=['Residues'],
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hoverinfo='skip',
        text=[residues_styled],
        texttemplate='%{text}',
        textfont={'size':14},
        xgap=0, ygap=0,
        zsmooth=False
    ), row=1, col=1)

    # Numerical rows: Use pastel_seq colorscale for predicted/binary/known, Blues for others
    for i, name in enumerate(data_rows.keys(), 2):
        colorscale = pastel_seq if name in ['Known Peptide', 'Binary Prediction', 'Predicted Score'] else 'Blues'
        fig.add_trace(go.Heatmap(
            z=[data_rows[name]],
            x=positions,
            y=[name],
            colorscale=colorscale,
            showscale=False,
            zsmooth='fast',
            hoverinfo='x+y+z',
            xgap=0, ygap=0
        ), row=i, col=1)

    # Secondary Structure (no smoothing)
    fig.add_trace(go.Heatmap(
        z=[ss_numeric],
        x=positions,
        y=['Secondary Structure'],
        colorscale=ss_colorscale,
        showscale=False,
        hoverinfo='text',
        text=[[ss_map_full.get(ss, 'Unknown') for ss in protein_df['SS'].fillna('-')]],
        xgap=0, ygap=0,
        zsmooth=False
    ), row=n_heatmap_rows-1, col=1)

    # Sites (no smoothing)
    fig.add_trace(go.Heatmap(
        z=[site_numeric],
        x=positions,
        y=['Sites'],
        colorscale=site_colorscale,
        showscale=False,
        hoverinfo='text',
        text=[[f'Site: {s}' for s in protein_df['sites']]],
        xgap=0, ygap=0,
        zsmooth=False
    ), row=n_heatmap_rows, col=1)

    # Add legend markers for Secondary Structure (outside subplot)
    for i, ss_code in enumerate(ss_codes):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=ss_colors[i]),
            name=f'SS: {ss_map_full[ss_code]} ({ss_code})',
            showlegend=True
        ))

    # Add legend markers for Site types
    for i, site_type in enumerate(site_types):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=site_colors_list[i % len(site_colors_list)]),
            name=f'Site: {site_type}',
            showlegend=True
        ))

    # Hide x-axis ticks except bottom heatmap row
    for r in range(1, n_heatmap_rows+1):
        show_ticks = (r == n_heatmap_rows)
        fig.update_xaxes(showticklabels=show_ticks, row=r, col=1)

    fig.update_layout(
        height=int(1200 * row_height * n_heatmap_rows),
        width=max(900, len(protein_df)*15),
        title_text=f"Protein Visualization — Gene: {protein_df['gene'].iloc[0]} | Accession: {protein_df['accession'].iloc[0]}",
        font=dict(size=11),
        margin=dict(t=100, b=120),  # bigger bottom margin for legend
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=-0.12,   # bring legend closer to plot
            xanchor="left",
            x=0,
            orientation="h",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11),
            traceorder="normal"
        )
    )

    return fig


st.title("Protein Interactive Visualization")

uploaded_file = st.file_uploader("Upload a ZIP file containing the data CSV", type=["zip"])

if uploaded_file is not None:
    df = load_data_from_zip(uploaded_file)

    genes_with_peptides = df.groupby('gene')['known_peptide'].max()
    genes_with_peptides = genes_with_peptides[genes_with_peptides == 1].index.tolist()

    selected_gene = st.selectbox("Select a gene", options=genes_with_peptides)

    protein_df = df[df['gene'] == selected_gene].copy()
    protein_df.reset_index(drop=True, inplace=True)

    threshold = st.slider("Prediction Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

    fig = interactive_protein_heatmap_with_sites(protein_df, threshold=threshold)

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a ZIP file containing the data CSV.")
