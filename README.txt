Hi there, I am Ose

# Streamlit minimal requirements
streamlit==1.22.0
pandas==1.5.3
plotly==5.13.0

# Gradio minimal requirements
gradio==3.36.1
pandas==1.5.3
plotly==5.13.0

# FastHTML minimal requirements
fasthtml==0.0.18
pandas==1.5.3
plotly==5.13.0

# app_streamlit.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Visualizer", layout="wide")
st.title("📊 Data Visualizer with Streamlit")

# Sidebar for inputs
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    st.header("Chart Settings")
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Scatter", "Line", "Bar", "Histogram"]
    )

# Use sample data if no file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = px.data.iris()
    st.info("Using sample Iris dataset. Upload a CSV file to use your own data.")

# Display data
st.header("Data Preview")
st.dataframe(df.head())

# Chart configuration
st.header("Chart Configuration")

col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X Axis", options=df.columns)
with col2:
    if chart_type != "Histogram":
        y_axis = st.selectbox("Y Axis", options=df.columns)
    else:
        y_axis = None

# Generate chart
st.header("Visualization")
if chart_type == "Scatter":
    fig = px.scatter(df, x=x_axis, y=y_axis, color=df.columns[0] if len(df.columns) > 0 else None)
elif chart_type == "Line":
    fig = px.line(df, x=x_axis, y=y_axis)
elif chart_type == "Bar":
    fig = px.bar(df, x=x_axis, y=y_axis)
elif chart_type == "Histogram":
    fig = px.histogram(df, x=x_axis)

st.plotly_chart(fig, use_container_width=True)

# Show data statistics
st.header("Data Statistics")
st.write(df.describe())


# app_gradio.py
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

def create_visualization(file, chart_type, x_axis, y_axis, use_sample_data):
    # Load data
    if use_sample_data or file is None:
        df = px.data.iris()
        data_source = "Sample Iris Data"
    else:
        df = pd.read_csv(file.name)
        data_source = "Uploaded Data"
    
    # Create visualization based on chart type
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis, 
                        title=f"{chart_type} Plot of {data_source}")
    elif chart_type == "Line":
        fig = px.line(df, x=x_axis, y=y_axis, 
                     title=f"{chart_type} Plot of {data_source}")
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, 
                    title=f"{chart_type} Plot of {data_source}")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, 
                          title=f"{chart_type} of {data_source}")
    
    # Convert plot to HTML
    plot_html = to_html(fig, include_plotlyjs="cdn", full_html=False)
    
    # Create data preview
    data_preview = df.head().to_html(classes="table table-striped")
    
    # Create statistics
    stats = df.describe().to_html(classes="table table-striped")
    
    return plot_html, data_preview, stats

# Load sample data to get column names
sample_df = px.data.iris()
columns = sample_df.columns.tolist()

# Create the Gradio interface
with gr.Blocks(title="Data Visualizer with Gradio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Data Visualizer with Gradio")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            use_sample_data = gr.Checkbox(label="Use Sample Data", value=True)
            chart_type = gr.Dropdown(
                choices=["Scatter", "Line", "Bar", "Histogram"],
                value="Scatter",
                label="Chart Type"
            )
            x_axis = gr.Dropdown(choices=columns, value=columns[0], label="X Axis")
            with gr.Row():
                y_axis = gr.Dropdown(choices=columns, value=columns[1], label="Y Axis")
            generate_btn = gr.Button("Generate Visualization", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("## Visualization")
            plot_output = gr.HTML(label="Plot")
            
            with gr.Tab("Data Preview"):
                data_preview = gr.HTML()
            
            with gr.Tab("Statistics"):
                stats_output = gr.HTML()
    
    # Set up event handling
    inputs = [file_input, chart_type, x_axis, y_axis, use_sample_data]
    generate_btn.click(create_visualization, inputs=inputs, outputs=[plot_output, data_preview, stats_output])
    
    # Update column options when use_sample_data is toggled or file is uploaded
    def update_columns(file, use_sample):
        if use_sample or file is None:
            df = px.data.iris()
        else:
            df = pd.read_csv(file.name)
        columns = df.columns.tolist()
        return gr.Dropdown(choices=columns, value=columns[0]), gr.Dropdown(choices=columns, value=columns[1] if len(columns) > 1 else columns[0])
    
    file_input.change(update_columns, inputs=[file_input, use_sample_data], outputs=[x_axis, y_axis])
    use_sample_data.change(update_columns, inputs=[file_input, use_sample_data], outputs=[x_axis, y_axis])

if __name__ == "__main__":
    demo.launch()





# app_fasthtml.py
from fasthtml import FastHTML, ui, app
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io
import base64

# Create app
app = FastHTML(title="Data Visualizer with FastHTML")

# Sample data
sample_df = px.data.iris()

# Store data in session
class SessionData:
    def __init__(self):
        self.df = sample_df
        self.chart_type = "Scatter"
        self.x_axis = sample_df.columns[0]
        self.y_axis = sample_df.columns[1]

sd = SessionData()

# Home page
@app.get
def home():
    return ui.H1("📊 Data Visualizer with FastHTML"), get_main_content()

def get_main_content():
    return ui.Div(
        ui.Form(
            ui.Div(
                ui.Label("Upload CSV File:", for_="file"),
                ui.Input(type="file", id="file", name="file", accept=".csv"),
                ui.Button("Upload", type="submit", name="action", value="upload", 
                         classes="btn btn-primary mt-2"),
                classes="mb-3"
            ),
            ui.Div(
                ui.Label("Or use sample data:", classes="me-2"),
                ui.Button("Load Sample Data", type="submit", name="action", 
                         value="sample", classes="btn btn-secondary"),
                classes="mb-3"
            ),
            ui.Div(
                ui.Label("Chart Type:", for_="chart_type"),
                ui.Select(
                    ui.Option("Scatter", value="Scatter", selected=True),
                    ui.Option("Line", value="Line"),
                    ui.Option("Bar", value="Bar"),
                    ui.Option("Histogram", value="Histogram"),
                    id="chart_type", name="chart_type", onchange="updateForm()"
                ),
                classes="mb-3"
            ),
            ui.Div(
                ui.Label("X Axis:", for_="x_axis"),
                ui.Select(
                    *[ui.Option(col, value=col, selected=(col == sd.x_axis)) 
                      for col in sd.df.columns],
                    id="x_axis", name="x_axis"
                ),
                classes="mb-3"
            ),
            ui.Div(
                ui.Label("Y Axis:", for_="y_axis"),
                ui.Select(
                    *[ui.Option(col, value=col, selected=(col == sd.y_axis)) 
                      for col in sd.df.columns],
                    id="y_axis", name="y_axis"
                ),
                id="y_axis_container",
                classes="mb-3"
            ),
            ui.Div(
                ui.Button("Generate Chart", type="submit", name="action", 
                         value="generate", classes="btn btn-success"),
                classes="mb-3"
            ),
            id="data-form",
            hx_post="/update",
            hx_target="#output",
            hx_swap="innerHTML"
        ),
        ui.Div(id="output"),
        ui.Script("""
            function updateForm() {
                const chartType = document.getElementById('chart_type').value;
                const yAxisContainer = document.getElementById('y_axis_container');
                
                if (chartType === 'Histogram') {
                    yAxisContainer.style.display = 'none';
                } else {
                    yAxisContainer.style.display = 'block';
                }
            }
            
            // Initialize form state
            document.addEventListener('DOMContentLoaded', function() {
                updateForm();
            });
        """)
    )

@app.post
def update(action: str, chart_type: str, x_axis: str, y_axis: str, file: ui.UploadFile = None):
    global sd
    
    if action == "upload" and file and file.filename:
        # Read uploaded file
        contents = file.file.read()
        sd.df = pd.read_csv(io.BytesIO(contents))
        sd.x_axis = sd.df.columns[0]
        sd.y_axis = sd.df.columns[1] if len(sd.df.columns) > 1 else sd.df.columns[0]
    
    elif action == "sample":
        # Use sample data
        sd.df = sample_df
        sd.x_axis = sample_df.columns[0]
        sd.y_axis = sample_df.columns[1]
    
    elif action == "generate":
        # Update chart settings
        sd.chart_type = chart_type
        sd.x_axis = x_axis
        sd.y_axis = y_axis
    
    # Generate the chart
    if sd.chart_type == "Scatter":
        fig = px.scatter(sd.df, x=sd.x_axis, y=sd.y_axis, 
                        title=f"{sd.chart_type} Plot")
    elif sd.chart_type == "Line":
        fig = px.line(sd.df, x=sd.x_axis, y=sd.y_axis, 
                     title=f"{sd.chart_type} Plot")
    elif sd.chart_type == "Bar":
        fig = px.bar(sd.df, x=sd.x_axis, y=sd.y_axis, 
                    title=f"{sd.chart_type} Plot")
    elif sd.chart_type == "Histogram":
        fig = px.histogram(sd.df, x=sd.x_axis, 
                          title=f"{sd.chart_type}")
    
    # Convert plot to HTML
    plot_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    
    # Create data preview
    data_preview = sd.df.head().to_html(classes="table table-striped", index=False)
    
    # Create statistics
    stats = sd.df.describe().to_html(classes="table table-striped")
    
    return ui.Div(
        ui.H2("Visualization"),
        ui.Raw(plot_html),
        ui.H2("Data Preview"),
        ui.Raw(data_preview),
        ui.H2("Data Statistics"),
        ui.Raw(stats)
    )

if __name__ == "__main__":
    app.run()


