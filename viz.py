# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:42:20 2018

This script will be used for visualizing TestData.csv from /DerivedData
@author: Cedric Anover
"""

# Note: I managed to run bokeh because I installed: conda install -c conda-forge pillow=4.0.0

import math
import pandas as pd
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import Viridis5
#from bokeh.core.properties import value
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

# Note: The only event here is the modifications/changes in TestData.csv

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
    
    def update():
        df_update = pd.read_csv(".\DerivedData\TestData.csv")
        df2_update = pd.DataFrame({'count': df_update.groupby("Job Category").size()}).reset_index()
        # x and y axes
        x_job_category_update = df2_update['Job Category'].tolist()
        y_count_update = df2_update['count'].tolist()
        
        new = {'x_job_category': x_job_category_update,
               'y_count': y_count_update}
        source.stream(new)

    doc.add_periodic_callback(update, 2000)
    
    # Bokeh's convenience function for creating a Figure object
    p = figure(x_range=x_job_category, 
               y_range=(0, max(y_count)+1000), 
               plot_height=1000,
               plot_width=2000,
               title="Count by Job Category",
               toolbar_location=None, #tools="pan,wheel_zoom,box_zoom,zoom_reset,previewsave"
               )
    
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


apps = {'/': Application(FunctionHandler(make_document))}
server = Server(apps, port=5001)
server.start()


#show(p)
###############################################################################
# Export the plot in html
#from bokeh.embed import file_html
#from bokeh.plotting import output_file, save
#output_file('job_category_viz.html', mode='inline')
#output_file('job_category_viz.html', mode='cdn')
#save(p)

###############################################################################
#from bokeh.io import curdoc
##from bokeh.layouts import column
#curdoc().add_root(p)

"""
To run the app on the server you will have to run it on the command line
bokeh serve --show viz.py
Technically, my computer is the physical server.
"""












