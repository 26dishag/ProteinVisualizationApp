import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
import zipfile

# Set Streamlit page layout to wide for fullscreen width
st.set_page_config(layout="wide")

@st.cache_data
def load_data_from_zip(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as csvfile:
            df = pd.read_csv(csvfile)
    return df

def expand_and_merge_ones(binary_array, merge_distance=2):
    arr = binary_array.copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == 1:
            j = i + 1
            while j < n and arr[j] == 0:
                j += 1
            gap_length = j - i - 1
            if j < n and 0 < gap_length <= merge_distance:
                arr[i+1:j] = 1
                i = j
            else:
                i = j
        else:
            i += 1
    return arr

def find_peptide_groups(predicted_binary, merge_distance=2, min_length=6):
    expanded = expand_and_merge_ones(predicted_binary, merge_distance=merge_distance)
    groups = []
    current_group = []

    for i, val in enumerate(expanded):
        if val == 1:
            current_group.append(i)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []
    if current_group:
        groups.append(current_group)

    peptide_group_labels = np.zeros(len(predicted_binary))
    for group in groups:
        if len(group) >= min_length:
            for pos in group:
                peptide_group_labels[pos] = 1
    return peptide_group_labels

def interactive_protein_heatmap_with_sites(protein_df):
    protein_df = protein_df.copy()

    # Auto-select threshold based on peptide group size
    possible_thresholds = np.linspace(0.4, 0.8, 20)
    selected_threshold = None
    for t in possible_thresholds:
        predicted_binary = (protein_df['predicted_score_bad'] > t).astype(int)
        peptide_groups = find_peptide_groups(predicted_binary.values, merge_distance=2, min_length=6)
        if 0 < peptide_groups.sum() < 0.15 * len(protein_df):
            selected_threshold = t
            break
    if selected_threshold is None:
        selected_threshold = 0.6  # fallback threshold

    threshold = selected_threshold
    protein_df['predicted_binary'] = (protein_df['predicted_score_bad'] > threshold).astype(int)
    peptide_groups = find_peptide_groups(protein_df['predicted_binary'].values, merge_distance=2, min_length=6)
    protein_df['peptide_group'] = peptide_groups

    residues = protein_df['AA'].tolist()
    positions = list(range(len(protein_df)))

    residues_styled = [
        f"<b><span style='color:red'>{res}</span></b>" if kp else res
        for res, kp in zip(residues, protein_df['known_peptide'].astype(bool))
    ]

    residues_peptide_styled = [
        f"<b><span style='color:blue'>{res}</span></b>" if pg else res
        for res, pg in zip(residues, protein_df['peptide_group'].astype(bool))
    ]

    def normalize(arr):
        arr = np.nan_to_num(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)

    data_rows = {
        'Predicted Score': normalize(protein_df['predicted_score_bad'].values),
        'Binary Prediction': normalize(protein_df['predicted_binary'].values),
        'Known Peptide': normalize(protein_df['known_peptide'].values),
        'Predicted Peptides': protein_df['peptide_group'].values,
        'Conservation': normalize(protein_df['conservation_norm'].fillna(0).values),
        'Pathogenicity': normalize(protein_df['pathogenicity'].fillna(0).values),
        'relASA': normalize(protein_df['relASA'].fillna(0).values),
    }

    sigma_horizontal = 0.2
    for key in ['Predicted Score', 'Binary Prediction', 'Conservation', 'Pathogenicity', 'relASA']:
        data_rows[key] = gaussian_filter1d(data_rows[key], sigma=sigma_horizontal)

    pastel_seq = [
        [0.0, '#fce4ec'], [0.2, '#f8bbd0'], [0.4, '#f48fb1'],
        [0.6, '#f06292'], [0.8, '#ec407a'], [1.0, '#e91e63']
    ]

    ss_map_full = {
        "H": "Alpha helix (4-12)", "B": "Isolated beta-bridge residue", "E": "Strand",
        "G": "3-10 helix", "I": "Pi helix", "T": "Turn", "S": "Bend", "P": "Kappa helix", "-": "Unknown"
    }
    ss_codes = list(ss_map_full.keys())
    ss_numeric = [ss_codes.index(ss) if ss in ss_codes else ss_codes.index('-') for ss in protein_df['SS'].fillna('-')]
    ss_colors = ['#a8dadc', '#457b9d', '#1d3557', '#f1faee', '#e63946', '#ffb703', '#fb8500', '#b56576', '#adb5bd']
    ss_colorscale = [[i / (len(ss_colors) - 1), c] for i, c in enumerate(ss_colors)]

    site_cols = [col for col in protein_df.columns if col.startswith('sites_')]
    protein_df['sites'] = protein_df[site_cols].idxmax(axis=1).str.replace('sites_', '')
    protein_df.loc[protein_df[site_cols].sum(axis=1) == 0, 'sites'] = 'None'
    site_types = sorted(protein_df['sites'].unique())
    site_map = {s: i for i, s in enumerate(site_types)}
    site_colors_list = ['#ffadad', '#ffd6a5', '#fdffb6', '#caffbf', '#9bf6ff', '#a0c4ff', '#bdb2ff', '#ffc6ff']
    site_colorscale = [[i / (len(site_types) - 1), site_colors_list[i % len(site_colors_list)]] for i in range(len(site_types))]
    site_numeric = [site_map[s] for s in protein_df['sites']]

    all_rows = ['Residues', 'Peptide Residues'] + list(data_rows.keys()) + ['Secondary Structure', 'Sites']
    n_heatmap_rows = len(all_rows)
    row_height = 0.05
    row_heights = [row_height] * n_heatmap_rows

    fig = make_subplots(
        rows=n_heatmap_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
        row_heights=row_heights,
        subplot_titles=[None] * n_heatmap_rows
    )

    # Residues row
    fig.add_trace(go.Heatmap(
        z=[[0] * len(positions)],
        x=positions,
        y=['Residues'],
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hoverinfo='skip',
        text=[residues_styled],
        texttemplate='%{text}',
        textfont={'size': 14},
        xgap=0, ygap=0,
        zsmooth=False
    ), row=1, col=1)

    # Peptide Residues row
    fig.add_trace(go.Heatmap(
        z=[[0] * len(positions)],
        x=positions,
        y=['Peptide Residues'],
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hoverinfo='skip',
        text=[residues_peptide_styled],
        texttemplate='%{text}',
        textfont={'size': 14},
        xgap=0, ygap=0,
        zsmooth=False
    ), row=2, col=1)

    # Data rows heatmaps
    for i, name in enumerate(data_rows.keys(), 3):
        colorscale = pastel_seq if name in ['Known Peptide', 'Binary Prediction', 'Predicted Score'] else 'Blues'
        if name == 'Predicted Peptides':
            colorscale = [[0, 'white'], [1, '#c2185b']]
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

    # Secondary structure heatmap
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
    ), row=n_heatmap_rows - 1, col=1)

    # Sites heatmap
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

    # X axis ticks only on bottom
    for r in range(1, n_heatmap_rows + 1):
        show_ticks = (r == n_heatmap_rows)
        fig.update_xaxes(showticklabels=show_ticks, row=r, col=1)

    # Secondary structure legend
    for ss_code, color in zip(ss_map_full.keys(), ss_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup='SS',
            showlegend=True,
            name=f"SS: {ss_map_full[ss_code]}"
        ))

    # Sites legend
    for site, color in zip(site_types, site_colors_list):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup='Sites',
            showlegend=True,
            name=f"Site: {site}"
        ))

    # Extract predicted peptide groups sequences
    peptide_groups_list = []
    current_peptide = []
    for idx, flag in enumerate(protein_df['peptide_group']):
        if flag == 1:
            current_peptide.append(protein_df['AA'].iloc[idx])
        else:
            if current_peptide:
                peptide_groups_list.append(''.join(current_peptide))
                current_peptide = []
    if current_peptide:
        peptide_groups_list.append(''.join(current_peptide))

    peptide_text = "<br>".join(peptide_groups_list) if peptide_groups_list else "None"

    # Add annotation with predicted ligands
    fig.add_annotation(
        text=f"<b>Predicted Ligands:</b><br>{peptide_text}",
        xref="paper", yref="paper",
        x=1.15, y=1,
        showarrow=False,
        font=dict(size=12),
        align="left",
        bordercolor="black",
        borderwidth=1,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )

    fig.update_layout(
        height=int(1200 * row_height * n_heatmap_rows),
        width=max(900, len(protein_df) * 15),
        margin=dict(t=100, b=150, r=400),
        title_text=f"Protein Visualization — Gene: {protein_df['gene'].iloc[0]} | Accession: {protein_df['accession'].iloc[0]} | Threshold: {threshold:.2f}",
        font=dict(size=11),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            orientation="h",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11)
        )
    )

    return fig

# Streamlit App starts here
st.title("Protein Interactive Visualization with Predicted Peptide Groups")

uploaded_file = st.file_uploader("Upload a ZIP file containing the data CSV", type=["zip"])

if uploaded_file is not None:
    df = load_data_from_zip(uploaded_file)

    # Filter genes with known peptides
    genes_with_peptides = df.groupby('gene')['known_peptide'].max()
    genes_with_peptides = genes_with_peptides[genes_with_peptides == 1].index.tolist()

    if not genes_with_peptides:
        st.warning("No genes with known peptides found in data.")
    else:
        selected_gene = st.selectbox("Select a gene", options=genes_with_peptides)
        protein_df = df[df['gene'] == selected_gene].copy().reset_index(drop=True)

        fig = interactive_protein_heatmap_with_sites(protein_df)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a ZIP file containing the data CSV.")
