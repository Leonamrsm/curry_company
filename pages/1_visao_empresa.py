# Libraries
import pandas as pd
import plotly.express as px
import folium
from haversine import haversine, Unit
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static


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

def order_metric(df):
    """
    Generate a bar chart showing the number of orders per day.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the order data.

    Returns:
        plotly.graph_objs._figure.Figure: The bar chart visualizing the number of orders per day.
    """
    cols = ['ID', 'Order_Date']

    # agrupamento
    df_aux = df.loc[:,cols].groupby('Order_Date').count().reset_index()

    # desenhar o gráfico de linhas
    fig = px.bar( df_aux, x='Order_Date', y='ID')

    return fig

def traffic_order_share(df):
    """
    Generate a pie chart showing the percentage of orders by road traffic density.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the order data.

    Returns:
        plotly.graph_objs._figure.Figure: The pie chart visualizing the percentage of orders by road traffic density.
    """
    df_aux = df.groupby('Road_traffic_density')['ID'].count().reset_index(name='count_of_orders')

    df_aux = df_aux[df_aux['Road_traffic_density'] != 'NaN']

    df_aux['perc_of_orders'] = round(100 * df_aux['count_of_orders'] / df_aux['count_of_orders'].sum(), 2)

    fig = px.pie(df_aux, values='perc_of_orders', names='Road_traffic_density', title='% de pedidos por tipo de transito')

    return fig

def traffic_order_city(df):
    """
    Generate a scatter plot showing the count of orders by city and road traffic density.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the order data.

    Returns:
        plotly.graph_objs._figure.Figure: The scatter plot visualizing the count of orders by city and 
        road traffic density.
    """
    df_aux = df.groupby(['City', 'Road_traffic_density'])['ID'].count().reset_index(name='count_of_orders')

    df_aux = df_aux.loc[ (df_aux['City']!= 'NaN') & (df_aux['Road_traffic_density']!= 'NaN'), :]

    fig = px.scatter(df_aux, x='City', y='Road_traffic_density', size='count_of_orders', color='City')

    return fig

def order_by_week_of_year(df):
    """
    Generate a line chart showing the number of orders by week.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the order data.

    Returns:
        plotly.graph_objs._figure.Figure: The line chart visualizing the number of orders by week.
    """

    # criar a coluna de semana
    df['week_of_year'] = df['Order_Date'].dt.strftime('%U')

    # colunas
    cols = ['ID', 'week_of_year']

    # agrupamento
    df_aux = df.loc[:,cols].groupby('week_of_year').count().reset_index()

    # desenhar o gráfico de linhas
    fig = px.line( df_aux, x='week_of_year', y='ID')

    return fig

def order_share_by_week(df):
    """
    Generate a line chart showing the order share by week, which is calculated by dividing the count 
    of orders by the number of unique delivery persons for each week.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the order data.

    Returns:
        plotly.graph_objs._figure.Figure: The line chart visualizing the order share by week.
    """

    # Quantidade de pedidos por semana / Número único de entregadores por semana
    df_aux1 = df1.groupby("week_of_year")["ID"].count().reset_index(name='count_orders_by_week')

    df_aux2 = df1.groupby("week_of_year")['Delivery_person_ID'].nunique().reset_index(name='number_of_unique_delivery_person')

    df_aux = pd.merge(df_aux1, df_aux2, how='inner')
    df_aux['order_by_unique_deliveries'] = df_aux.count_orders_by_week/df_aux.number_of_unique_delivery_person

    fig = px.line(df_aux, x='week_of_year', y='order_by_unique_deliveries')

    return fig

def country_maps(df1):
    """
    Generate a map of with markers showing the median latitude and longitude of 
    restaurants based on the city and road traffic density.

    Parameters:
    - df1 (pandas.DataFrame): The input DataFrame containing the restaurant data.

    Returns:
    - None
    """
    df_aux = (df1.groupby(['City', 'Road_traffic_density'])[['Restaurant_latitude', 'Restaurant_longitude']]
                .median()
                .reset_index())

    df_aux = df_aux[df_aux['City'] != 'NaN']

    map = folium.Map()

    for index, location_info in df_aux.iterrows():
        folium.Marker([location_info['Restaurant_latitude'], 
                    location_info['Restaurant_longitude']],
                    popup=location_info[['City', 'Road_traffic_density']]).add_to(map)
    
    folium_static(map, width=1024, height=600)

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

st.header('Marketplace - Visão Cliente', divider='rainbow')

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

with tab1:
    with st.container():
        st.markdown('# Orders by Day')
        
        fig = order_metric(df1)

        st.plotly_chart(fig, use_container_width=True)

        with st.container():   

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('# Traffic Order Share')
                fig = traffic_order_share(df1)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('# Traffic Order City')
                fig = traffic_order_city(df1)
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("# Orders by Week")

    with st.container():
        fig = order_by_week_of_year(df1)
        st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.markdown("# Order Share by Week")
        fig = order_share_by_week(df1)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("# Country Maps")
    country_maps(df1)
