from ctypes import alignment
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)

phase_colors = {'PreIdle': '#636EFA', 
          'Working': '#EF553B', 
          'PostIdle': '#00CC96'} 

data_path = './data/phase_gantt.csv'
res_df = pd.read_csv(data_path, header=0).set_index('Index')

timeline_fig = px.timeline(res_df, x_start='Start', x_end='Finish', y='Task', color='Task', 
    color_discrete_map=phase_colors, title='Timeline of Reading Process')

res_df = res_df.iloc[1:-1]
phase_time_avg = res_df.groupby('Task')['Time'].mean().reset_index()
pie_fig = px.pie(phase_time_avg, values='Time', names='Task', color='Task', color_discrete_map=phase_colors, title='Average phase time of book reading')
pie_fig.update_traces(hoverinfo='label+percent', texttemplate="%{value:.3f}s <br>(%{percent})")

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Cami is a FOOL!', style={'textAlign': 'center'}),

    html.Div(children='''
        PhaseCut: Analysis of Phase Cut Result.
    ''', style={'textAlign': 'center'}),

    dcc.Graph(
        id='timeline-graph',
        figure=timeline_fig
    ),
    dcc.Graph(
        id='pie-graph',
        figure=pie_fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)