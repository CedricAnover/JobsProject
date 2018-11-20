import pandas as pd
import numpy as np
import math

import bokeh
from bokeh.core.properties import value
from bokeh.palettes import Viridis
from bokeh.io import show
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, FactorRange, HoverTool
from bokeh.transform import dodge
from bokeh.transform import factor_cmap


#import holoviews as hv
#import hvplot.pandas

#hv.extension('bokeh', 'matplotlib', width="100")

# Get Data from source
seek_jobs = pd.read_csv(".\RawData\SeekJobs.csv", encoding = "ISO-8859-1")

# Clean the Date Column
seek_jobs['Date'] = pd.to_datetime(seek_jobs['Date'], format="%d-%b-%y") #  "%d-%b-%Y"

# Create new DataFrame from seek_jobs groupby 'Date' & 'Department'
df2 = pd.DataFrame({'count': seek_jobs.groupby(["Date", "Department"]).size()}).reset_index()

# Create new DataFrame pivoting df2 where index='Date', columns='Department', values='count'
df3 = df2.pivot(index='Date', columns='Department', values='count')

# Convert all elements in df3 from float to int
df3 = df3.fillna(0).astype(int)
#df3 = df3.astype('int64',errors='ignore')

# Treat index of df3 as a column 'Date'
df3['Date'] = df3.index

# Convert datetimes to strings
df3['Date'] = df3['Date'].dt.strftime('%Y-%m-%d')

# Create new DataFrame. Make sure that the index is not a Date type but a string
df4 = df3.loc[:, df3.columns != 'Date']
df4.index = df4.index.strftime('%Y-%m-%d')  # Convert to string

# Create a new DataFrame, transposing df4
df5 = df4.T
###############################################################################
# Source: https://bokeh.pydata.org/en/latest/docs/gallery/bar_nested_colormapped.html
departments = list(df5.index)
dates = list(df5.columns)
pall = Viridis[256]

x = [(date, department) for date in dates for department in departments]
counts = sum(zip(*[df5[col] for col in df5.columns]), ())
#counts = tuple(map(lambda x: int(x) if x is float else x,counts))
source = ColumnDataSource(data=dict(x=x, counts=counts))

TOOLTIPS = [
    ("Count", "@counts"),
    ("(Date, Job Category)","@x")
]

#TOOLTIPS = [
#    ("Count", "@counts"),
#    ("Date","@x$sx"),
#    ("Job Category","@x$sy")
#]

p = figure(x_range=FactorRange(*x),
           y_range=(0, max(counts)+10),
           plot_height=1000, 
           plot_width=1500,
           title="Job Category count by Dates",
           toolbar_location="right", 
           tools="pan,wheel_zoom,box_zoom,reset,save",
           tooltips=TOOLTIPS)

p.xaxis.major_label_orientation = math.pi/4
#p.xaxis.minor_label_orientation = math.pi/4

p.vbar(x='x', 
       top='counts', 
       width=0.9, 
       source=source, 
       line_color="black")

#p.y_range.start = 0
#p.x_range.range_padding = 0.1
#p.xaxis.major_label_orientation = 1
#p.xgrid.grid_line_color = None

show(p)

###############################################################################
# Source https://stackoverflow.com/questions/47301262/how-to-plot-a-group-by-dataframe-in-bokeh-as-bar-chart
#data = df3.to_dict(orient='list')

#x_departments = list(data.keys())
#try:
#    x_departments.remove('Date')
#except ValueError:
#    pass
#dates = df3['Date'].tolist()
#source = ColumnDataSource(data=data)
#x = [(date, department) for date in dates for department in x_departments]
#pall = Viridis[256]

#p = figure(x_range=x, 
#           y_range=(0, df3[df3.columns.difference(['Date'])].values.max() + 3), 
#           plot_height=500, 
#           title="Count by Department and Date",
#           toolbar_location=None, 
#           tools="")

#for dep in df3.columns:
#    p.vbar(x=dodge('Date', -0.25, range=p.x_range), 
#           top=dep, 
#           width=0.4, 
#           source=source,
#           color="#c9d9d3", 
#           legend=value(dep))
#del dep

#p.vbar(x=dodge('Date', -0.25, range=p.x_range),
#       top=dep,
#       width=0.4,
#       source=source,
#       palette=pall,
#       legend=value(dep))



#p.xaxis.major_label_orientation = math.pi/4
#p.yaxis.major_label_orientation = "vertical"

#p.x_range.range_padding = 0.1
#p.xgrid.grid_line_color = None
#p.legend.location = "top_left"
#p.legend.orientation = "horizontal"

#show(p)

###############################################################################
#output_file("seek_viz.html")

#df2.loc[dates,x_departments].hvplot.bar('Date', by='Department', rot=90)
