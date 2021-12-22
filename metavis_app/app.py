# http://dash-bootstrap-components.opensource.faculty.ai/examples/iris/

import flask
from dash import Dash, Input, Output, State, callback
from dash import dcc, html
from dash import dash_table as dt
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_cytoscape as cyto

from datetime import datetime
import pandas as pd
import numpy as np
import re

server = flask.Flask(__name__)
app = Dash(external_stylesheets=[dbc.themes.MINTY], server=server)

######################################### Data Loading #########################################

metadata_sim_mat = np.load('similarities_matrix.npy')
tag_sim_mat = np.load('tag_distance_matrix.npy')
embedding_sim_mat = np.load('embedding_similarities.npy')
uuid_topic_pairing = pd.read_csv('lda_df.csv', usecols=['data_uuid', 'topic'])
column_signature_df = pd.read_csv('column_signature_df.csv')

df = pd.read_csv('data_catelog.csv', lineterminator='\n')
df['Recommended Dataset'] = df['data_name'].copy()
df['Downloads'] = df['download_count'].copy()
df['latest_update_year'] = df['latest_update_time'].apply(lambda x: int(x.split('-')[0]))
data_name_to_data_uuid_mapping = df.set_index('data_name')['data_uuid'].to_dict()
data_uuid_to_data_name_mapping = df.set_index('data_uuid')['data_name'].to_dict()
data_uuid_to_dataset_index_mapping = df.set_index('data_uuid')['dataset_index'].to_dict()
data_uuid_to_data_category_class_mapping = df.set_index('data_uuid')['data_category'].apply(lambda x: x.lower().replace('& ', '').replace(' ', '_')).to_dict()
data_uuid_to_asset_type_mapping = df.set_index('data_uuid')['asset_type'].to_dict()


def get_link_column_info(dataset_pair, column_signature_df):
    pair_df = column_signature_df[column_signature_df['data_uuid'].isin(dataset_pair)].copy()
    temp_agg = pair_df.groupby(['name_cleaned', 'column_data_type', 'cached_cardinality', 'cached_top_standardized'])['column_uuid'].apply(list)
    column_uuid_pair_list = temp_agg[temp_agg.apply(len) > 1].tolist()
    link_column_info = [list(pair_df.loc[pair_df['column_uuid'].isin(column_uuid_pair), ['column_uuid', 'data_uuid', 'field_name']].T.to_dict().values()) for column_uuid_pair in column_uuid_pair_list] if len(column_uuid_pair_list) > 0 else []
    return link_column_info


def get_short_data_name(data_uuid, max_length=30):
    full_data_name = data_uuid_to_data_name_mapping[data_uuid]
    short_data_name = (full_data_name[:max_length - 3] + '...') if len(full_data_name) > max_length else full_data_name
    return short_data_name

def recommend_datasets(data_uuid, embedding_weight=0.6, tag_weight=0.3, metadata_weight=0.1, return_uuids=False, return_k_uuids=5):

    rank_df = df[['data_uuid', 'dataset_index', 'data_name']].copy()
    mask = df['dataset_index'].tolist()

    dataset_index = data_uuid_to_dataset_index_mapping[data_uuid]

    # Metadata Features
    rank_df['metadata_similarity'] = metadata_sim_mat[dataset_index][mask]
    rank_df['metadata_rank'] = rank_df['metadata_similarity'].rank(method='min', ascending=False)
    # Tag
    rank_df['tag_similarity'] = tag_sim_mat[dataset_index][mask]
    rank_df['tag_similarity'] = rank_df['tag_similarity'] / (rank_df['tag_similarity'].max())
    rank_df['tag_similarity'] = rank_df['tag_similarity'].fillna(0)
    rank_df['tag_rank'] = rank_df['tag_similarity'].rank(method='min', ascending=False)
    # Embedding
    rank_df['embedding_similarity'] = embedding_sim_mat[dataset_index][mask]
    rank_df['embedding_rank'] = rank_df['embedding_similarity'].rank(method='min', ascending=False)
    # Weighted rank
    rank_df['weighted_rank'] = embedding_weight * rank_df['embedding_rank'] + tag_weight * rank_df['tag_rank'] + metadata_weight * rank_df['metadata_rank']

    rank_df.loc[rank_df['data_uuid'] == data_uuid, 'weighted_rank'] = 0
    rank_df = rank_df.sort_values(by=['weighted_rank'])

    if return_uuids:
        return rank_df['data_uuid'].head(1 + return_k_uuids).tolist()

    rank_df = rank_df.merge(uuid_topic_pairing)
    rank_df = rank_df.merge(df)

    return rank_df[:100]

######################################### Search Box #########################################


search_box_label = dbc.Label("Source Dataset",
                             style={'fontSize': '16', 'width': '100px', 'height': '30px', 'verticalAlign': 'top', 'marginTop': '6px', 'display': 'inline-block'}
                             )

dataset_name_options = df['Recommended Dataset'].value_counts().index.tolist()
search_box = dcc.Dropdown(id="dataset_search_box",
                          options=[
                              {"label": x, "value": x} for x in dataset_name_options
                          ],
                          style={'width': 'calc(100% - 100px)', 'height': '30px', 'verticalAlign': 'top', 'display': 'inline-block'}
                          )

search_section = html.Div([search_box_label,
                           search_box],
                          style={'fontSize': 12},
                          )

######################################### Filters #########################################

#### latest_update_time_filter ####
latest_update_time_label = dbc.Label("Latest Update Time", style={'width': '100%', 'height': '10px', 'verticalAlign': 'bottom'})
latest_update_time_filter = dcc.RangeSlider(
    id="latest_update_time_filter",
    min=2011,
    max=2021,
    value=[2011, 2021],
    marks={num: {'label': str(num), 'style': {'fontSize': 9}} for num in range(2011, 2021 + 1, 2)},
    allowCross=False,
    tooltip={"placement": "bottom", "always_visible": False},
)

#### update_frequency_filter ####
update_frequency_label = dbc.Label("Update Frequency", style={'width': '100%', 'height': '10px', 'verticalAlign': 'bottom'})
update_frequency_options = ['Daily', 'Monthly', 'Annually', 'One-time']
update_frequency_filter = dcc.Dropdown(
    id="update_frequency_filter",
    options=[
        {"label": x, "value": x} for x in update_frequency_options
    ],
    value=update_frequency_options,
    multi=True,
    style={'width': '100%', 'height': '30px', 'verticalAlign': 'bottom', 'display': 'inline-block'}
)

#### data_owner_filter ####
data_owner_label = dbc.Label("Data Owner", style={'width': '100%', 'height': '10px', 'verticalAlign': 'bottom'})
data_owner_options = df['data_owner'].value_counts().index.tolist()
data_owner_filter = dcc.Dropdown(
    id="data_owner_filter",
    options=[
        {"label": x, "value": x} for x in data_owner_options
    ],
    value=data_owner_options,
    style={'width': '100%', 'height': '30px', 'verticalAlign': 'bottom', 'display': 'inline-block'}
)

#### ai_topic_filter ####
ai_topic_label = dbc.Label("AI Topic Filter", style={'width': '100%', 'height': '10px', 'verticalAlign': 'bottom'})
ai_topic_filter = daq.BooleanSwitch(id='ai_topic_filter', on=False, disabled = True)

############################################

filter_row_1 = dbc.Row(
    [
        dbc.Col([update_frequency_label, update_frequency_filter], md=7),
        dbc.Col([latest_update_time_label, latest_update_time_filter], md=5),
    ]
)

filter_row_2 = dbc.Row(
    [
        dbc.Col([data_owner_label, data_owner_filter], md=7),
        dbc.Col([ai_topic_label, ai_topic_filter], md=5),
    ]
)

######################################### RCMD Table #########################################

rcmd_table_section = html.Div([
    dt.DataTable(
        id='datatable',
        data=[],
        columns=[{'name': c, 'id': c} for c in ['Recommended Dataset', 'Downloads', 'Add / Remove']],
        fixed_rows={'headers': True},
        page_action='none',
        style_table={'height': '360px', 'overflowY': 'auto'},  # 240px
        style_cell={
            'textAlign': 'left', 'fontSize': 12, 'whiteSpace': 'normal'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Recommended Dataset'},
             'minWidth': '400px', 'width': '400px', 'maxWidth': '400px'},
            {'if': {'column_id': 'Downloads'},
             'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
            {'if': {'column_id': 'Add / Remove'},
             'minWidth': '50px', 'width': '50px', 'maxWidth': '50px', 'textAlign': 'center'},
        ],
        style_data_conditional=[
            {'if': {'filter_query': '{Add / Remove} = "+"', 'column_id': 'Add / Remove'},
             'fontSize': '12', 'fontWeight': 'bold', 'color': 'green'},
            {'if': {'filter_query': '{Add / Remove} = "-"', 'column_id': 'Add / Remove'},
                'fontSize': '12', 'fontWeight': 'bold', 'color': 'red'},
            {"if": {"state": "active"},
                "backgroundColor": "rgba(0, 0, 0, 0)", "border": "1px solid rgb(211,211,211)"},
        ],
        style_as_list_view=True,
    )
]
)

######################################### Left Panel #########################################

left_panel = dbc.Card(
    [
        search_section,
        html.Hr(),
        filter_row_1,
        filter_row_2,
        html.Hr(),
        rcmd_table_section
    ],
    style={'fontSize': 10},
    body=True,
)

#######################################  network_graph ###############################################

network_graph = cyto.Cytoscape(
        id='network_vis',
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '300px'},
        elements=[
        ],
        stylesheet=[
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'width': 20,
                    'height': 20,
                    'font-size': 6,
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'line-style': 'dashed',
                }
            },
            # Class selectors
            {
                'selector': '.selected',
                'style': {
                    'border-width': '2px',
                    'border-color': 'blue'
                }
            },
            {
                'selector': '.map',
                'style': {
                    'shape': 'roundrectangle'
                }
            },
            {
                'selector': '.link_available',
                'style': {
                    'line-style': 'solid',
                }
            },
            {
                'selector': '.missing',
                'style': {
                    'background-color': '#a9a9a9'
                }
            },
            {
                'selector': '.education',
                'style': {
                    'background-color': '#3cb44b'
                }
            },
            {
                'selector': '.city_government',
                'style': {
                    'background-color': '#f1c232'
                }
            },
            {
                'selector': '.social_services',
                'style': {
                    'background-color': '#f58231'
                }
            },
            {
                'selector': '.transportation',
                'style': {
                    'background-color': '#911eb4'
                }
            },
            {
                'selector': '.housing_development',
                'style': {
                    'background-color': '#42d4f4'
                }
            },
            {
                'selector': '.environment',
                'style': {
                    'background-color': '#f032e6'
                }
            },
            {
                'selector': '.public_safety',
                'style': {
                    'background-color': '#bfef45'
                }
            },
            {
                'selector': '.health',
                'style': {
                    'background-color': '#9A6324'
                }
            },
            {
                'selector': '.business',
                'style': {
                    'background-color': '#800000'
                }
            },
            {
                'selector': '.recreation',
                'style': {
                    'background-color': '#808000'
                }
            },
            {
                'selector': '.nyc_bigapps',
                'style': {
                    'background-color': '#000075'
                }
            },

        ]
    )


####################################### More Info Tabs #######################################

##################### metadata_tab #####################

def replace_html_link_with_markdown_link(x):
    return re.sub(r'<a href="(.*?)">(.*?)<\/a>',r'[\2](\1)',x)

def clip_text(x, length):
    clipped_text = ((x[:length][::-1]).split(maxsplit=1)[1])[::-1]
    return (clipped_text+(' ...' if not clipped_text.endswith('.') else '')) if len(x)>length else x

def get_more_dataset_info_markdown(data_uuid):
    info_dict = df[df['data_uuid'] == data_uuid].reset_index().T.to_dict()[0]
    data_name = info_dict['data_name']
    data_name = clip_text(data_name, 40)
    data_category = info_dict['data_category']
    data_owner = info_dict['data_owner']
    asset_type = info_dict['asset_type']
    readable_size = info_dict['readable_size']
    update_frequency = info_dict['update_frequency'].lower()
    latest_update_time = info_dict['latest_update_time']
    description = info_dict['description']
    description = clip_text(replace_html_link_with_markdown_link(description), 300)
    data_page_url = info_dict['data_page_url']
    more_dataset_info_markdown = f"""#### {data_name}  \n**Category**: {data_category} | **Owner**: {data_owner}  \nThis is a {readable_size} {asset_type} that is updated {update_frequency}, latest update on {latest_update_time}.  \n **Description**  \n{description}  \nFor more information, visit the [official dataset page]({data_page_url})."""
    return more_dataset_info_markdown

more_dataset_info_markdown = ''
metadata_tab_content = dcc.Markdown(more_dataset_info_markdown,
            id='more_dataset_info',
            style={'width': '100%','height': '100%', 'overflow': 'auto', 'paddingTop':'10px'}
            )

##################### linkage_tab #####################

linkage_code_markdown = ''
linkage_code_part = html.Div([
        dcc.Markdown(
            linkage_code_markdown,
            id='linkage_code',
            style={'width': '100%', 'height': 200, 'overflow': 'auto'},
        ),
        dcc.Clipboard(
            target_id='linkage_code',
            style={
                'position': 'absolute',
                'top': 0,
                'right': 20,
                'fontSize': '15px',
            },
        )
    ],
    style={'width': '100%', 'height': 200, 'position': 'relative'}
)

linkage_code_explanation = dcc.Markdown(
            'Based on the edge you selected, we have generated code to download and merge the two datasets. You can register and get your NYC Open Data API token [here](https://data.cityofnewyork.us/profile/edit/developer_settings).',
            style={'width': '100%', 'overflow': 'auto', 'paddingTop':'10px'},
        )

linkage_tab_content = html.Div([
    linkage_code_explanation,
    linkage_code_part
    ])

##################### combine_tabs #####################

more_info_tabs = dbc.Tabs(
            [
                dbc.Tab(children=metadata_tab_content, label="Metadata", tab_id="tab-1", id="metadata_tab"),
                dbc.Tab(children=linkage_tab_content, label="Linkage", tab_id="tab-2", id="linkage_tab"),
            ],
            id="more_info_tabs",
            active_tab="tab-1"
        )

######################################### Right Panel #########################################

right_panel = html.Div(
    [
        html.Div(
            [
                network_graph
            ],
        ),
        html.Div(
            [
                more_info_tabs
            ],
        ),
    ],
    style={'height': '100%'},
)

######################################### Whole Page Layout #########################################

datasets_in_network_vis_dict = dict(zip(df['data_uuid'].tolist(), [False] * len(df)))
datasets_in_network_vis_dict['latest_added'] = None

app.layout = dbc.Container(
    [
        html.H1("MetaVis Dashboard", style={'paddingTop':'10px'}),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(left_panel, md=6),
                dbc.Col(right_panel, md=6),
            ],
            align="start",
        ),
        dcc.Store(id='recommendations'),
        dcc.Store(id='filtered_recommendations'),
        dcc.Store(id='datasets_in_network_vis', data=datasets_in_network_vis_dict),
    ],
    fluid=True,
)

######################################### Callbacks #########################################
#############################################################################################

####################### update_recommendations #######################
@ app.callback(
    Output('recommendations', 'data'),
    Input('dataset_search_box', 'value')
)
def update_recommendations(data_name):
    print('FUNC: update_recommendations')
    if isinstance(data_name, str):
        data_uuid = data_name_to_data_uuid_mapping.get(data_name, None)
        if isinstance(data_uuid, str):
            return recommend_datasets(data_uuid).to_dict()
    return df.to_dict()

####################### update_filtered_recommendations #######################
@ app.callback(
    Output('filtered_recommendations', 'data'),
    Input('recommendations', 'data'),
    Input('datasets_in_network_vis', 'data'),
    Input('latest_update_time_filter', 'value'),
    Input('update_frequency_filter', 'value'),
    Input('data_owner_filter', 'value'),
    Input('ai_topic_filter', 'on'),
)
def update_filtered_recommendations(rcmd_df_dict, datasets_in_network_vis, latest_update_time_range, update_frequency_list, data_owner_value, ai_topic_on_or_not):
    print('FUNC: update_filtered_recommendations')
    rcmd_df = pd.DataFrame.from_dict(rcmd_df_dict)
    rcmd_df['TRUE'] = True

    start_year, end_year = map(int, latest_update_time_range)
    latest_update_time_filter_series = (rcmd_df['latest_update_year'] >= start_year) & (rcmd_df['latest_update_year'] <= end_year)

    update_frequency_filter_series = (rcmd_df['update_frequency'].isin(update_frequency_list))

    data_owner_filter_series = (rcmd_df['data_owner'] == data_owner_value) if (isinstance(data_owner_value, str) and data_owner_value in data_owner_options) else rcmd_df['TRUE']

    ai_topic_filter_series = (rcmd_df['topic'] == rcmd_df['topic'].head(1).tolist()[0]) if ai_topic_on_or_not == True else rcmd_df['TRUE']

    filtered_rcmd_df = rcmd_df[latest_update_time_filter_series & update_frequency_filter_series & data_owner_filter_series & ai_topic_filter_series].copy()

    filtered_rcmd_df['Add / Remove'] = filtered_rcmd_df['data_uuid'].map(datasets_in_network_vis).apply(lambda x: '-' if x else '+')

    return filtered_rcmd_df.to_dict()

####################### enable_ai_topic_filter #######################
@ app.callback(
    Output('ai_topic_filter', 'disabled'),
    Input('filtered_recommendations', 'data')
)
def enable_ai_topic_filter(filtered_rcmd_df_dict):
    print('FUNC: enable_ai_topic_filter')
    filtered_rcmd_df = pd.DataFrame.from_dict(filtered_rcmd_df_dict)
    if len(filtered_rcmd_df) == len(df):
        return True
    else:
        return False


####################### populate_datatable #######################
@ app.callback(
    Output('datatable', 'data'),
    Input('filtered_recommendations', 'data'),
)
def populate_datatable(filtered_rcmd_df_dict):
    print('FUNC: populate_datatable')
    filtered_rcmd_df = pd.DataFrame.from_dict(filtered_rcmd_df_dict)
    return filtered_rcmd_df[['Recommended Dataset', 'Downloads', 'Add / Remove']].to_dict('records')

####################### update_based_on_active_cell #######################

@ app.callback(
    Output('datasets_in_network_vis', 'data'),
    Input('datatable', 'active_cell'),
    State('filtered_recommendations', 'data'),
    State('datasets_in_network_vis', 'data')
)
def update_based_on_active_cell(active_cell, filtered_rcmd_df_dict, datasets_in_network_vis):
    print('FUNC: update_based_on_active_cell')
    filtered_rcmd_df = pd.DataFrame.from_dict(filtered_rcmd_df_dict).reset_index(drop=True)
    if isinstance(active_cell, dict) and active_cell['column_id'] == 'Add / Remove':
        selected_data_uuid = filtered_rcmd_df.loc[active_cell['row'], 'data_uuid']
        datasets_in_network_vis[selected_data_uuid] = not datasets_in_network_vis[selected_data_uuid]
        if datasets_in_network_vis[selected_data_uuid] == True:
            datasets_in_network_vis['latest_added'] = selected_data_uuid
        else:
            datasets_in_network_vis['latest_added'] = None
    return datasets_in_network_vis

####################### update_network_vis_element #######################


@ app.callback(
    Output('network_vis', 'elements'),
    Input('datasets_in_network_vis', 'data'),
    State('network_vis', 'elements')
)
def update_network_vis_element(datasets_in_network_vis, network_vis_current_elements):
    print('FUNC: update_network_vis_element')
    seed_datasets = [k for k, v in datasets_in_network_vis.items() if v and k!='latest_added']

    nodes = [{'data': {'id': data_uuid, 'label': get_short_data_name(data_uuid)},
              'classes': 'node selected ' + data_uuid_to_data_category_class_mapping[data_uuid] + (' map' if data_uuid_to_asset_type_mapping[data_uuid] == 'map' else '')} for data_uuid in seed_datasets]
    node_id_list = [dic['data']['id'] for dic in nodes]
    edges = []
    edge_id_list = []
    for seed_data_uuid in seed_datasets:
        suggested_datasets = recommend_datasets(seed_data_uuid, return_uuids=True)[1:]  # first returned is the seed itself
        for suggested_data_uuid in suggested_datasets:
            if suggested_data_uuid not in seed_datasets:
                nodes.append({'data': {'id': suggested_data_uuid, 'label': get_short_data_name(suggested_data_uuid)}, 'classes': 'node suggested ' + data_uuid_to_data_category_class_mapping[suggested_data_uuid] + (' map' if data_uuid_to_asset_type_mapping[suggested_data_uuid] == 'map' else '')})
                node_id_list.append(suggested_data_uuid)
            temp_source, temp_target = sorted([seed_data_uuid, suggested_data_uuid])
            edge_link_data = get_link_column_info((temp_source, temp_target), column_signature_df)
            edge_id = '--'.join([temp_source, temp_target])
            if edge_id not in edge_id_list:
                edges.append({'data': {'id': edge_id, 'source': temp_source, 'target': temp_target, 'link_data': edge_link_data}, 'classes': 'edge' + (' link_available' if len(edge_link_data) > 0 else '')})
    elements = nodes + edges
    return elements


####################### update_metadata_tab #######################
@app.callback(Output('more_dataset_info', 'children'),
              Input('network_vis', 'tapNodeData'),
              Input('datasets_in_network_vis', 'data'),
              State('more_dataset_info', 'children'))  #$$$$
def update_metadata_tab(clicked_node, datasets_in_network_vis, current_markdown):
    print('FUNC: update_metadata_tab')
    current_showing_data_uuid = current_markdown.split('/')[-1].replace(').','')
    if clicked_node and clicked_node['id'] != current_showing_data_uuid:
        more_dataset_info_markdown = get_more_dataset_info_markdown(clicked_node['id'])
        return more_dataset_info_markdown
    elif datasets_in_network_vis['latest_added'] != None:
        more_dataset_info_markdown = get_more_dataset_info_markdown(datasets_in_network_vis['latest_added'])
        return more_dataset_info_markdown
    else:
        return ''

####################### update_linkage_tab #######################
@app.callback(Output('linkage_code', 'children'),
              Input('network_vis', 'tapEdgeData'))
def update_linkage_tab(data):
    print('FUNC: update_linkage_tab')
    if data:
        data_uuid_1 = data['source']
        data_name_1 = data_uuid_to_data_name_mapping[data_uuid_1]
        data_uuid_2 = data['target']
        data_name_2 = data_uuid_to_data_name_mapping[data_uuid_2]
        link_data = data['link_data']
        if len(link_data)>0:
            notice = ''
            left_on_columns = [[dic['field_name'] for dic in li if dic['data_uuid']==data_uuid_1][0] for li in link_data]
            right_on_columns = [[dic['field_name'] for dic in li if dic['data_uuid']==data_uuid_2][0] for li in link_data]
            left_on_columns_string = ','.join(["'"+col+"'" for col in left_on_columns])
            right_on_columns_string = ','.join(["'"+col+"'" for col in right_on_columns])
            merge_command = f"df = pd.merge(df1, df2, left_on=[{left_on_columns_string}], right_on=[{right_on_columns_string}])"
        else:
            notice = '# Directly mergeable columns not detected, experiment how to merge yourself.'
            merge_command = ''
        linkage_code_markdown = f"""```
{notice}
import pandas as pd
from sodapy import Socrata
client = Socrata('data.cityofnewyork.us', YOUR_API_TOKEN)
def fetch_data(data_uuid, nrows = None, usecols = [], chunksize = 5000):
    record_length = int(client.get(data_uuid, select="COUNT(*)")[0]['COUNT'])
    nrows = min(0 if nrows is None else nrows, record_length)
    start = 0
    results =[]
    while start <= nrows:
      results.extend( client.get(data_uuid, offset=start, limit=chunksize))
      start = start + chunksize
    return pd.DataFrame.from_records(results)
df1 = fetch_data('{data_uuid_1}') # {data_name_1}
df2 = fetch_data('{data_uuid_2}') # {data_name_2}
{merge_command}
```"""
        return linkage_code_markdown
    else:
        pass

####################### update_active_tab_choice #######################
@app.callback(Output("more_info_tabs", 'active_tab'),
    Input('network_vis', 'tapEdgeData'),
    Input('network_vis', 'tapNodeData'),
    Input('datasets_in_network_vis', 'data'),
    State('linkage_code', 'children'),
    State('more_dataset_info', 'children'),
    )
def update_active_tab_choice(clicked_edge, clicked_node, datasets_in_network_vis, linkage_code, current_markdown):
    print('FUNC: update_active_tab_choice')

    ## Still buggy...
    # print(clicked_node, clicked_edge, datasets_in_network_vis['latest_added'])

    linkage_code = '' if linkage_code == None else linkage_code
    current_displaying_node = current_markdown.split('/')[-1].replace(').','')
    current_displaying_edge = tuple(re.findall(r"fetch_data\('(.*?)'\)",linkage_code))

    clicked_node_change, clicked_edge_change = False, False
    if clicked_node:
        clicked_node_change = clicked_node['id'] != current_displaying_node
    if clicked_edge:
        clicked_edge_change = (clicked_edge['source'],clicked_edge['target']) != current_displaying_edge

    # print('clicked_node_change:',clicked_node_change,'clicked_edge_change',clicked_edge_change)
    # print('current_displaying_node',current_displaying_node,'current_displaying_edge',current_displaying_edge)

    if not clicked_node_change and not clicked_edge_change:
        return 'tab-1'
    elif clicked_node_change:
        return 'tab-1'
    else:
        return 'tab-2'

######################################### Driver #########################################

if __name__ == "__main__":
    app.run_server(debug=True)

