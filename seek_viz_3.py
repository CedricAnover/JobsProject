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


output_file("seek_viz3.html")
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

# Date range variable
df_date_range = (df2['Date'].min(), df2['Date'].max())

# For each job category, we create a filtered pd.DataFrame and store it in job_cat_dict dictionary
job_cat_dict = {}
for job in departments:
    job_cat_dict[job] = df2.loc[df2['Department'] == job]
    job_cat_dict[job].reset_index(inplace=True, drop=True)
    job_cat_dict[job].set_index('Date', inplace=True)
    
del job

# Create Empty DataFrame with Date index from df_date_range
df3 = pd.DataFrame(index=pd.date_range(start=df_date_range[0], end=df_date_range[1]))
df3['count'] = None

#df3.join(job_cat_dict[job], lsuffix='_caller', rsuffix='_other')
for k,v in job_cat_dict.items():
    df3 = df3.join(v[['count']], rsuffix=f" {k}")
df3 = df3.loc[:, df3.columns != 'count']
del k,v

# Create a list of dates
dates = df3.index.tolist()

# Make the index in df3 as a column
df3.reset_index(inplace=True)
df3.rename(columns={'index':'Dates'}, inplace=True) # Rename 'Index' to 'Dates'

source = ColumnDataSource(df3)

###############################################################################
# Create the data to be used for bokeh's ColumnDataSource

def make_fig(title, xname, yname):
    """Return a figure object"""
    TOOLTIPS = [
            ("Count", "@yname"),
            ("Date","@xname")
            ]
    
    p = figure(title = title,
               y_range = (0,df3[yname].max()+5),
               x_range = DataRange1d(start=df_date_range[0],end=df_date_range[1]),
               x_axis_label="Date",
               x_axis_type='datetime',
               y_axis_label="Count",
               toolbar_location="right",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               tooltips=TOOLTIPS)
    
    p.xaxis.major_label_orientation = math.pi/4
    
    p.xaxis.formatter=DatetimeTickFormatter(
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
    
    p.vbar(x=xname,
           top=yname,
           source=source,
           width=10)
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
figs = [make_fig(jobcategory,"Dates",f"count {jobcategory}") for jobcategory in departments]


# Create a grid with 4 columns
grid = gridplot(list(batch(figs,3)),
                toolbar_location="right")

#del grid

#from bokeh.layouts import column
#show(column(*figs))

show(grid)

#example_fig = make_fig("Aerospace Engineering","x Aerospace Engineering","y Aerospace Engineering")
#del example_fig

###############################################################################
"""
What I'm going to do is a bit similar to the following source
https://bokeh.pydata.org/en/latest/docs/gallery/anscombe.html
except that we have multiple categories and we'll use bar chart.
"""





































