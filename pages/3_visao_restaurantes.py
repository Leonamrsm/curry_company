import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from haversine import haversine, Unit
import plotly.graph_objs as go
import numpy as np

# ==============================================================================
# Funções
# ==============================================================================

def clean_code(df1):
    """
    Cleans the given DataFrame by removing rows with missing values in specific columns and converting
    data types of certain columns.
    
    Parameters:
        df1 (pandas.DataFrame): The DataFrame to be cleaned.
        
    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """

    linhas_selecionadas = ((df1['Delivery_person_Age'].str.strip() == 'NaN') | 
                        (df1['multiple_deliveries'].str.strip() == 'NaN') | 
                        (df1['City'].str.strip() == 'NaN') |
                        (df1['Festival'].str.strip() == 'NaN') |
                        (df1['Road_traffic_density'].str.strip() == 'NaN'))

    df1 = df1.loc[~linhas_selecionadas, :].copy()

    # 1. Convertendo Delivery_person_Age para int
    df1["Delivery_person_Age"] = df1["Delivery_person_Age"].astype(int)

    # 2. Convertendo Delivery_person_Ratings para float
    df1["Delivery_person_Ratings"] = df1["Delivery_person_Ratings"].astype(float)

    # 3. Convertendo Order_Date para datetime
    df1["Order_Date"] = pd.to_datetime(df1["Order_Date"], format="%d-%m-%Y")

    # 4. Convertendo multiple_deliveries para int
    df1["multiple_deliveries"] = df1["multiple_deliveries"].astype(int)

    # 5. Removendo os espaços dentro de strings/texto/object
    for col in df1.select_dtypes(include=['object']).columns:
        df1[col] = df1[col].str.strip()

    # 6. Limpando a coluna 'Time_taken(min)'
    df1['Time_taken(min)'] = df1['Time_taken(min)'].apply(lambda x: x.split(' ')[1]).astype(int)

    return df1

def distance(df):
    """
    Calculates the average distance in kilometers between each restaurant and the delivery location.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the latitude and longitude of the 
        restaurant and the delivery location.

    Returns:
        float: The average distance in kilometers.
    """
    def calculate_distance(row): 

        coords_1 = (row['Restaurant_latitude'], row['Restaurant_longitude'])
        coords_2 = (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
        return haversine(coords_1, coords_2, unit=Unit.KILOMETERS)

    df['Distance_km'] = df.apply(calculate_distance, axis=1)

    return df['Distance_km'].mean().round(2)

def avg_std_time_taken_by_Festival(df):
    """
    Calculate the average and standard deviation of the 'Time_taken(min)' field grouped by 'Festival'.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the 'Time_taken(min)' field and the 'Festival' field.

    Returns:
        pandas.DataFrame: A DataFrame with the 'Festival' field and two new fields: 'Time_taken(min)_mean' and 'Time_taken(min)_std'.
    """
    # Agrupamento e agregação
    df_aux = df.groupby('Festival').agg({'Time_taken(min)' : ['mean', 'std']})

    # Ajustar os nomes das colunas
    df_aux.columns = ['Time_taken(min)_mean', 'Time_taken(min)_std']

    # Resetar o índice
    df_aux = df_aux.reset_index()

    return df_aux

def bar_plot_time_by_city(df):
    """
    Generates a bar plot showing the average and standard deviation of the 'Time_taken(min)' field grouped by 'City'.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the 'Time_taken(min)' field and the 'City' field.

    Returns:
        plotly.graph_objects.Figure: A bar plot with the 'City' field on the x-axis and the average 
        'Time_taken(min)' on the y-axis, with error bars representing the standard deviation.
    """
    df_aux = df.groupby('City').agg({'Time_taken(min)' : ['mean', 'std']}).reset_index()
    df_aux.columns = ['City', 'Time_taken(min)_mean', 'Time_taken(min)_std']
    df_aux = df_aux.loc[df_aux['City'] != 'NaN',:]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Control', 
                        x=df_aux['City'],
                        y=df_aux['Time_taken(min)_mean'],
                        error_y=dict(type='data', array=df_aux['Time_taken(min)_std'])))

    fig.update_layout(barmode='group')

    return fig

def avg_std_time_by_traffic(df1):
    """
    Calculates the average and standard deviation of the 'Time_taken(min)' field grouped by 'City' 
    and 'Road_traffic_density'.

    Parameters:
        df1 (pandas.DataFrame): The input DataFrame containing the 'Time_taken(min)' field, the 'City' 
        field, and the 'Road_traffic_density' field.

    Returns:
        pandas.DataFrame: A DataFrame with the 'City', 'Road_traffic_density', 'Time_taken(min)_mean', 
        and 'Time_taken(min)_std' columns.
    """
    # Agrupamento e agregação
    df_aux = df1.groupby(['City', 'Road_traffic_density']).agg({'Time_taken(min)': ['mean', 'std']})

    # Ajustar os nomes das colunas
    df_aux.columns = ['Time_taken(min)_mean', 'Time_taken(min)_std']

    # Resetar o índice
    df_aux = df_aux.reset_index()

    return df_aux

def pie_plot_avg_distance_by_city(df):
    """
    Generates a pie plot showing the average distance by city.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the 'City' and 'Distance_km' columns.

    Returns:
        plotly.graph_objects.Figure: A pie plot showing the average distance by city.
    """

    avg_distance = df.loc[:, ['City', 'Distance_km']].groupby('City').mean().reset_index()

    fig = go.Figure(data=[go.Pie(labels=avg_distance['City'], values=avg_distance['Distance_km'], pull=[0, 0.1, 0])])

    return fig

def sunburst_plot(df1):
    """
    Generates a sunburst plot based on the input DataFrame.

    Parameters:
        df1 (pandas.DataFrame): The input DataFrame containing 'City', 'Road_traffic_density', 
        and 'Time_taken(min)' columns.

    Returns:
        plotly.graph_objects.Figure: A sunburst plot showing the mean and standard deviation of 
        'Time_taken(min)' grouped by 'City' and 'Road_traffic_density'.
    """
    # Agrupamento e agregação
    df_aux = df1.groupby(['City', 'Road_traffic_density']).agg({'Time_taken(min)': ['mean', 'std']})

    # Ajustar os nomes das colunas
    df_aux.columns = ['Time_taken(min)_mean', 'Time_taken(min)_std']

    # Resetar o índice
    df_aux = df_aux.reset_index()

    fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'], values='Time_taken(min)_mean', 
        color='Time_taken(min)_std', color_continuous_scale='RdBu', 
        color_continuous_midpoint=np.average(df_aux['Time_taken(min)_std']))
    
    return fig

# ==============================================================================
# Beginning of the Logical Structure of the Code
# ==============================================================================

# Import Dataset
df = pd.read_csv("dataset/train.csv")

df1 = clean_code(df)

# ==============================================================================
# Sidebar
# ==============================================================================

image_path = 'logo.png'
image = Image.open(image_path)
st.sidebar.image(image, width=270)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""___""")

st.sidebar.markdown('## Selecione uma data limite')

# Definindo os valores mínimo e máximo para o slider
min_date = df1['Order_Date'].min().date()
max_date = df1['Order_Date'].max().date()

date_slider = st.sidebar.slider(
    'Limit date',
    min_value=min_date,
    max_value=max_date,
    value=max_date,
    format='DD-MM-YYYY'
)

st.sidebar.markdown("""___""")

traffic_options = st.sidebar.multiselect(
    'What are the traffic conditions?',
    df1['Road_traffic_density'].unique(),
    default=df1['Road_traffic_density'].unique()
)    

st.sidebar.markdown("""___""")

st.sidebar.markdown('### Powered by Leonam Rezende')

# filtering b max Order_Date
df1 = df1.loc[df1['Order_Date'].dt.date < date_slider,:]

# filtering by Traffic Conditions
df1 = df1.loc[df1['Road_traffic_density'].isin(traffic_options),:]

# ==============================================================================
# Sreamlit Layout
# ==============================================================================

st.header('Marketplace - Visão Restaurantes', divider='rainbow')



tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        df_aux = avg_std_time_taken_by_Festival(df1)
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            col1.metric(label="Unique Deliveries", value=df1['Delivery_person_ID'].nunique())

        with col2:
            avg_distance = distance(df1)
            col2.metric(label="Mean Distance", value=avg_distance)

        with col3:
            col3.metric(label = 'AVG T Fest', value=df_aux[df_aux['Festival'] =='Yes']['Time_taken(min)_mean'].round(2))

        with col4:
            col4.metric(label = 'STD T Fest', value=df_aux[df_aux['Festival'] =='Yes']['Time_taken(min)_std'].round(2))

        with col5:
            col5.metric(label = 'AVG T NO Fest', value=df_aux[df_aux['Festival'] =='No']['Time_taken(min)_mean'].round(2))

        with col6:
            col6.metric(label = 'STD T No Fest', value=df_aux[df_aux['Festival'] =='No']['Time_taken(min)_std'].round(2))

        st.markdown("""___""")

    with st.container():

        col1, col2 = st.columns(2)

        with col1:

            st.markdown('##### Average Time in minutes')
            fig = bar_plot_time_by_city(df1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('##### The average delivery time and standard deviation by city and traffic type.')
            df_aux = avg_std_time_by_traffic(df1)
            st.dataframe(df_aux)

        st.markdown("""___""")

    with st.container():
        st.title('Time Distribution')
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('##### Average Delivery Time by City')
            fig = pie_plot_avg_distance_by_city(df1)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('##### Average Speed')
            fig = sunburst_plot(df1)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""___""")
