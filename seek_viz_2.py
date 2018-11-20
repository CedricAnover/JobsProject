import pandas as pd
import numpy as np
import math

import bokeh
from bokeh.core.properties import value
from bokeh.palettes import Viridis
from bokeh.io import show
from bokeh.plotting import figure, output_file
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, DatetimeTickFormatter
from bokeh.transform import dodge
from bokeh.transform import factor_cmap
from bokeh.models.ranges import DataRange1d

from bokeh.util.browser import view
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import column, gridplot
from bokeh.models import Circle, ColumnDataSource, Div, Grid, Line, LinearAxis, Plot, Range1d
from bokeh.resources import INLINE


output_file("seek_viz2.html")
###############################################################################
#### Data Pre-processing
# Get raw data
seek_jobs = pd.read_csv(".\RawData\SeekJobs.csv", encoding = "ISO-8859-1")

# Clean the Date Column. 
seek_jobs['Date'] = pd.to_datetime(seek_jobs['Date'], format="%d-%b-%y")  # If we only want date, then add .dt.date

# Transform Date column into String
#seek_jobs['Date'] = seek_jobs['Date'].dt.strftime('%Y-%m-%d')

# Create new DataFrame from seek_jobs groupby 'Date' & 'Department'
df2 = pd.DataFrame({'count': seek_jobs.groupby(["Date", "Department"]).size()}).reset_index()

# Create a list containing all unique Job Categories
departments = list(set([val for val in df2["Department"]]))

# For each job category, we create a filtered pd.DataFrame and store it in job_cat_dict dictionary
job_cat_dict = {}
for job in departments:
    job_cat_dict[job] = df2.loc[df2['Department'] == job]
    job_cat_dict[job].reset_index(inplace=True, drop=True)
del job

# Fill the date gaps
#for k,v in job_cat_dict.items():
#    pass
#del k,v
#job_cat_dict['Aerospace Engineering']['Date'].dt.strftime('%Y-%m-%d')
#dir(job_cat_dict['Aerospace Engineering']['Date'][0])
###############################################################################
# Create the data to be used for bokeh's ColumnDataSource
data = {}
for k,v in job_cat_dict.items():
    data['x {}'.format(k)] = v['Date'].tolist()
    data['y {}'.format(k)] = v['count'].tolist()
del k,v
source = ColumnDataSource(data)
"""Example
data = {'x': [1,2,3,4], 'y': np.ndarray([10.0, 20.0, 30.0, 40.0])}
source = ColumnDataSource(data)
"""

def make_fig(title, xname, yname):
    """Return a figure object"""
    TOOLTIPS = [
    ("Count", "@yname"),
    ("Date","@xname")
    ]
    
    p = figure(title = title,
               y_range = (0,df2['count'].mean()),
               x_range = DataRange1d(start=df2['Date'].min(),end=df2['Date'].max()),
               x_axis_label="Date",
               x_axis_type='datetime',
               y_axis_label="Count",
               toolbar_location="right",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               tooltips=TOOLTIPS)

#    p = figure(title = title, 
#               x_axis_label="Date",
#               y_axis_label="Count",
#               toolbar_location="right",
#               tools="pan,wheel_zoom,box_zoom,reset,save")
    
    p.xaxis.major_label_orientation = math.pi/4
    
    p.xaxis.formatter=DatetimeTickFormatter(
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
    
    p.vbar(x=xname,
           top=yname,
           source=source,
           width=0.9)
    return p

def batch(iterable, n):
    counter = 0
    current_batch = []
    for element in iterable:
        current_batch.append(element)
        counter +=1
        if counter % n == 0:
            yield current_batch
            current_batch = []
    yield current_batch

# Create a list of figures
figs = [make_fig(jobcategory,f"x {jobcategory}",f"y {jobcategory}") for jobcategory in ['Automotive Trades']]


# Create a grid with 4 columns
#grid = gridplot(list(batch(figs,4)),
#                toolbar_location="right")

#del grid

#from bokeh.layouts import column
show(column(*figs))




#show(grid)



#example_fig = make_fig("Aerospace Engineering","x Aerospace Engineering","y Aerospace Engineering")
#del example_fig

###############################################################################
"""
What I'm going to do is a bit similar to the following source
https://bokeh.pydata.org/en/latest/docs/gallery/anscombe.html
except that we have multiple categories and we'll use bar chart.
"""





































