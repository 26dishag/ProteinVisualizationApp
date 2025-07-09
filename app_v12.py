import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
import zipfile

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# CSS override to make main container full width
st.markdown(
    """
    <style>
    .main > div {
        max-width: 100% !important;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True
)

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

def underline_special_residues(protein_df):
    n = len(protein_df)
    underline = np.zeros(n, dtype=bool)

    protein_df = protein_df.reset_index(drop=True)

    dibasic_indices = protein_df.index[protein_df['sites'].str.contains('dibasic', case=False, na=False)].tolist()
    cysteine_indices = protein_df.index[protein_df['sites'].str.contains('cysteine', case=False, na=False)].tolist()

    ss_flags = protein_df.get('signal_peptide_or_Strand', pd.Series(np.zeros(n))).values

    for idx in dibasic_indices:
        underline[idx] = True
        # backward extension
        for i in range(idx - 1, max(idx - 11, -1), -1):
            if i < 0 or ss_flags[i] == 1 or (i in dibasic_indices and i != idx):
                break
            underline[i] = True
        # forward extension
        for i in range(idx + 1, min(idx + 11, n)):
            if ss_flags[i] == 1 or (i in dibasic_indices and i != idx):
                break
            underline[i] = True

    if cysteine_indices:
        first_cys = cysteine_indices[0]
        last_cys = cysteine_indices[-1]

        if first_cys <= 9:
            underline[0:first_cys + 1] = True
        else:
            underline[0:10] = True

        if last_cys >= n - 10:
            underline[last_cys:n] = True
        else:
            underline[max(n - 10, 0):n] = True

    return underline

def interactive_protein_heatmap_with_sites(protein_df, threshold, smoothing_sigma):
    protein_df = protein_df.copy()
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

    if smoothing_sigma > 0:
        smoothed_binary = gaussian_filter1d(data_rows['Binary Prediction'], sigma=smoothing_sigma)
    else:
        smoothed_binary = data_rows['Binary Prediction']

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

    # Add the Peptide Hotspots mask for special residues
    underline_mask = underline_special_residues(protein_df)

    # Because we added 1 row for Peptide Hotspots, increase heatmap rows count by 1
    all_rows = [
        'Residues', 'Peptide Residues', 'Peptide Hotspots', 'Predicted Score', 'Smoothed Binary Prediction',
        'Known Peptide', 'Predicted Peptides', 'Conservation', 'Pathogenicity', 'relASA',
        'Secondary Structure', 'Sites', 'Signal Peptide/Strand'
    ]
    n_heatmap_rows = len(all_rows)
    row_height = 0.05
    row_heights = [row_height, row_height, 0.01] + [row_height] * (n_heatmap_rows - 4) + [0.01]

    fig = make_subplots(
        rows=n_heatmap_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
        row_heights=row_heights,
        subplot_titles=[None] * n_heatmap_rows
    )

    # Residues (row 1)
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

    # Peptide Residues (row 2)
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

    # Peptide Hotspots (row 3) - NEW
    fig.add_trace(go.Heatmap(
        z=[underline_mask.astype(int)],
        x=positions,
        y=['Peptide Hotspots'],
        colorscale=[[0, 'white'], [1, '#254373']],  # Dark blue fill for hotspots
        showscale=False,
        hoverinfo='x+y+z',
        xgap=0, ygap=0,
        zsmooth=False
    ), row=3, col=1)

    # Predicted Score (row 4, shifted from 3)
    fig.add_trace(go.Heatmap(
        z=[data_rows['Predicted Score']],
        x=positions,
        y=['Predicted Score'],
        colorscale=pastel_seq,
        showscale=False,
        zsmooth='fast',
        hoverinfo='x+y+z',
        xgap=0, ygap=0
    ), row=4, col=1)

    # Smoothed Binary Prediction (row 5, shifted from 4)
    fig.add_trace(go.Heatmap(
        z=[smoothed_binary],
        x=positions,
        y=['Smoothed Binary Prediction'],
        colorscale=pastel_seq,
        showscale=False,
        zsmooth='fast',
        hoverinfo='x+y+z',
        xgap=0, ygap=0
    ), row=5, col=1)

    # Update other rows in row_mapping, shifted by +1
    row_mapping = {
        'Known Peptide': 6,
        'Predicted Peptides': 7,
        'Conservation': 8,
        'Pathogenicity': 9,
        'relASA': 10
    }

    for name, row in row_mapping.items():
        colorscale = pastel_seq if name in ['Known Peptide', 'Predicted Score'] else 'Blues'
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
        ), row=row, col=1)

    # Secondary Structure (row 11, shifted from 10)
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
    ), row=11, col=1)

    # Sites (row 12, shifted from 11)
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
    ), row=12, col=1)

    # Signal Peptide/Strand (row 13, shifted from 12)
    fig.add_trace(go.Heatmap(
        z=[protein_df['signal_peptide_or_Strand'].fillna(0).values],
        x=positions,
        y=['Signal Peptide/Strand'],
        colorscale=[[0, 'white'], [1, '#6a4c93']],
        showscale=False,
        hoverinfo='text',
        text=[[f"Signal: {int(val)}" for val in protein_df['signal_peptide_or_Strand'].fillna(0).values]],
        xgap=0, ygap=0,
        zsmooth=False
    ), row=13, col=1)

    for r in range(1, n_heatmap_rows + 1):
        show_ticks = (r == n_heatmap_rows)
        fig.update_xaxes(showticklabels=show_ticks, row=r, col=1)

    for ss_code, color in zip(ss_map_full.keys(), ss_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup='SS',
            showlegend=True,
            name=f"SS: {ss_map_full[ss_code]}"
        ))

    for site, color in zip(site_types, site_colors_list):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup='Sites',
            showlegend=True,
            name=f"Site: {site}"
        ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='#6a4c93'),
        legendgroup='Signal',
        showlegend=True,
        name="Signal Peptide/Strand"
    ))

    fig.update_layout(
        autosize=True,
        width=None,
        height=int(1200 * row_height * n_heatmap_rows),
        margin=dict(t=100, b=150, r=100),
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

    return fig, protein_df

# Streamlit App
st.title("Protein Interactive Visualization with Predicted Peptide Groups")

uploaded_file = st.file_uploader("Upload a ZIP file containing the data CSV", type=["zip"])

if uploaded_file is not None:
    df = load_data_from_zip(uploaded_file)

    genes_with_peptides = df.groupby('gene')['known_peptide'].max()
    genes_with_peptides = genes_with_peptides[genes_with_peptides == 1].index.tolist()

    if not genes_with_peptides:
        st.warning("No genes with known peptides found in data.")
    else:
        selected_gene = st.selectbox("Select a gene", options=genes_with_peptides)
        protein_df = df[df['gene'] == selected_gene].copy().reset_index(drop=True)

        threshold = st.slider(
            "Select threshold for prediction",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01
        )

        smoothing_sigma = st.slider(
            "Select smoothing sigma for binary prediction (0 = no smoothing)",
            min_value=0.0,
            max_value=3.0,
            value=0.0,
            step=0.1
        )

        fig, processed_df = interactive_protein_heatmap_with_sites(protein_df, threshold, smoothing_sigma)

        peptide_groups_list = []
        current_peptide = []
        for idx, flag in enumerate(processed_df['peptide_group']):
            if flag == 1:
                current_peptide.append(processed_df['AA'].iloc[idx])
            else:
                if current_peptide:
                    peptide_groups_list.append(''.join(current_peptide))
                    current_peptide = []
        if current_peptide:
            peptide_groups_list.append(''.join(current_peptide))

        peptide_count = len(peptide_groups_list)
        peptide_text = "<br>".join(peptide_groups_list) if peptide_groups_list else "None"

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### Predicted Ligands (Count: {peptide_count})")
        st.markdown(peptide_text, unsafe_allow_html=True)

else:
    st.info("Please upload a ZIP file containing the data CSV.")
