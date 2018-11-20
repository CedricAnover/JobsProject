from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource

import math
import pandas as pd
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import Viridis5

def make_document(doc):
    # Load the data from \DerivedData\TestData.csv
    df = pd.read_csv(".\DerivedData\TestData.csv")
    df2 = pd.DataFrame({'count': df.groupby("Job Category").size()}).reset_index()
    
    # Source: https://stackoverflow.com/questions/47910177/bar-chart-showing-count-of-each-category-month-wise-using-bokeh
    
    # x and y axes
    x_job_category = df2['Job Category'].tolist()
    y_count = df2['count'].tolist()
    
    # Bokeh's mapping of column names and data lists
    source = ColumnDataSource(data=dict(x_job_category=x_job_category, y_count=y_count, color=Viridis5))
    
    # Bokeh's convenience function for creating a Figure object
    p = figure(x_range=x_job_category, 
               y_range=(0, max(y_count)+1000), 
               plot_height=1000,
               plot_width=2000,
               title="Count by Job Category",
               toolbar_location=None, 
               tools="")
    
    # Render and show the vbar plot
    p.vbar(x='x_job_category', 
           top='y_count', 
           width=2.0, 
           color='color', 
           source=source)
    
    # Source: https://stackoverflow.com/questions/39401481/how-to-add-data-labels-to-a-bar-chart-in-bokeh
    
    labels = LabelSet(x='x_job_category', 
                      y='y_count', 
                      text='y_count', 
                      level='glyph',
                      source=source, 
                      render_mode='canvas')
    p.add_layout(labels)
    
    p.xaxis.major_label_orientation = math.pi/4
    p.yaxis.major_label_orientation = "vertical"
    doc.add_root(p)
    show(p)

apps = {'/': Application(FunctionHandler(make_document))}

server = Server(apps, port=5000)
server.start()


# localhost:5000
# localhost:5001