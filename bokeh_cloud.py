import pickle
import pandas 
from bokeh.plotting import figure, ColumnDataSource, output_notebook, output_file, show, save, curdoc, vplot 
from bokeh.models import Circle, HoverTool, WheelZoomTool, PanTool, BoxZoomTool, ResetTool, TapTool, SaveTool, Slider, Button, DataTable, TableColumn, NumberFormatter,Dropdown
from bokeh.palettes import grey
from bokeh.layouts import layout, widgetbox, row, gridplot

path =''
data_all = pickle.load(open(path + 'data_all.p', 'rb'))

category = 'cluster_km'


mi = min(data_all[category])
ma = max(data_all[category])
i = mi
category_items = data_all[category].unique()

palette = grey(len(category_items))
colormap = dict(zip(category_items, palette))
colormap[i] =  '#FDE724'
color = data_all[category].map(colormap)
source_all = ColumnDataSource(data=dict(x = data_all['Component 1']
                                    , y = data_all['Component 2']
                                    , label = data_all['sentence'] 
                                    , color = color
                                    , size = data_all['cood_no']))
    
source2 = ColumnDataSource(data=dict())
current = data_all[(data_all[category] == i) ]#& (df['salary'] <= slider.value[1])].dropna() 
source2.data = {'sentence' : current.sentence } 
    ######################################################################### Scatter 
p = figure(plot_width=600
           , plot_height=600
           , title="2D Cluster Visualisation"
           , tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave" #hover,
           , x_axis_type=None
           , y_axis_type=None
           , min_border=1)

scatter = Circle(x = 'x'
                      , y = 'y'
                      #, marker = "circle"
                      #, source = source 
                      , fill_color = "color"
                      , line_color = "color"
                      , fill_alpha = 0.5
                      , size = 'size')

p.add_glyph(source_all, scatter)

hover = p.select(dict(type=HoverTool))
hover.tooltips={"cluster_km":"@label"}

def update_color(attrname, old, new):
        i = clust.value
        colormap = dict(zip(category_items, palette))
        colormap[i] = '#FDE724'
        color = data_all[category].map(colormap)
        source_all.data = dict(x = data_all['Component 1']
                                        , y = data_all['Component 2']
                                        , label = data_all['sentence'] 
                                        , color = color
                                        , size = data_all['cood_no'])
        p.update()

clust = Slider(title="cluster", value=mi, start=mi, end=ma, step=1)    
clust.on_change('value', update_color)
######################################################################### Scatter end
######################################################################### Table

def update(attrname, old, new): 
    i = clust.value
    current = data_all[(data_all[category] == i) ]#& (df['salary'] <= slider.value[1])].dropna() 
    source2.data = {'sentence': current.sentence} 

columns = [ 
     TableColumn(field="sentence", title="feedback")
] 
data_table = DataTable(source=source2, columns=columns, width=600)
table = widgetbox(data_table) 

clust.on_change('value', update)
######################################################################### Table end

def update_model(attrname, old, new):
    if dropdown.value == 'km': category = 'cluster_km'
    elif dropdown.value == 'ward': category = 'cluster_ward'
    elif dropdown.value == 'dbscan': category = 'cluster_dbscan'
    mi = min(data_all[category])
    ma = max(data_all[category])
    i = mi
    current = data_all[(data_all[category] == i) ]
    source2.data = {'sentence': current.sentence} 
    colormap = dict(zip(category_items, palette))
    colormap[i] = '#FDE724'
    color = data_all[category].map(colormap)
    source_all.data = dict(x = data_all['Component 1']
                                        , y = data_all['Component 2']
                                        , label = data_all['sentence'] 
                                        , color = color
                                        , size = data_all['cood_no'])
    #update_color()
    #p.update()
    
menu = [("km", "km"), ("ward", "ward"), ("dbscan", "dbscan")]
dropdown = Dropdown(label="Cluster model", menu=menu) # button_type="warning",
dropdown.on_change('value', update_model)
#dropdown.on_change('value', update_color)
#dropdown.on_change('value', update)


# put the button and plot in a layout and add to the document
controls = widgetbox(clust,dropdown)
layout = layout([controls],[p,table], sizing_mode='scale_width')
curdoc().add_root(layout)
curdoc().title = str(category)