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


def get_avg_ratings_by(df, col):
    """
    Calculates the average ratings and standard deviation of delivery person ratings 
    for each category in the given column.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        col (str, optional): The column to group by. 

    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - col (str): The category.
            - col+'_mean' (float): The average rating of delivery person ratings.
            - col + '_std' (float): The standard deviation of delivery person ratings.
    """


    df_aux = (df.groupby(col)
            .agg({'Delivery_person_Ratings' : ['mean', 'std']})
            .reset_index())

    df_aux = df_aux[df_aux[col] != 'NaN']
    df_aux.columns = [col, col +'_mean', col + '_std']
    
    return df_aux

def get_top_delivery_person_by_City(df1, fastest=True):
    """
    Returns a DataFrame with the top 10 delivery persons for each city, sorted by average time taken.
    
    Parameters:
        df1 (pandas.DataFrame): A DataFrame containing order data with columns 'City', 'Delivery_person_ID', and 'Time_taken(min)'.
        fastest (bool, optional): A boolean indicating whether to sort by fastest delivery time. Defaults to True.
    
    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - City (str): The name of the city.
            - Delivery_person_ID (str): The ID of the delivery person.
            - avg_time (float): The average time taken by the delivery person.
    """


    df_aux = df1.groupby(['City','Delivery_person_ID'])['Time_taken(min)'].mean().reset_index(name='avg_time')
    df_aux = df_aux[df_aux['City'] != 'NaN']
    df_aux = df_aux.sort_values(by=['City', 'avg_time'], ascending=[True, fastest]).groupby('City').head(10).reset_index(drop=True)

    return df_aux


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

st.header('Marketplace - Visão Entegradores', divider='rainbow')

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '', ''])

with tab1:
    with st.container():
        st.title('Overall Metrics')

        col1, col2, col3, col4 = st.columns(4, gap='large')

        with col1:
            col1.metric(label = 'Biggest Age', value=df1['Delivery_person_Age'].max())

        with col2:
            col2.metric(label = 'Lowest Age',  value=df1['Delivery_person_Age'].min())

        with col3:
            col3.metric(label = 'Best Vehicle Condition', value=df1['Vehicle_condition'].max())

        with col4:
            col4.metric(label = 'Worst Vehicle Condition', value=df1['Vehicle_condition'].min())

    st.markdown("""___""")

    with st.container():

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('##### Average Ratings by Delivery Person')

            df_aux = get_avg_ratings_by(df1, 'Delivery_person_ID')
            st.dataframe(df_aux)
        
        with col2:

            with st.container():

                st.markdown('##### Average Rating by Type of Traffic')
                df_aux = get_avg_ratings_by(df1, 'Road_traffic_density')
                st.dataframe(df_aux)

            with st.container():

                st.markdown('##### Average Rating by Weather')
                df_aux = get_avg_ratings_by(df1, 'Weatherconditions')

                st.dataframe(df_aux)

        st.markdown("""___""")

    with st.container():

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('##### Fastest Delivery Person by City')
            df_aux = get_top_delivery_person_by_City(df1, True)
            st.dataframe(df_aux)
        
        with col2:
            st.markdown('##### Slowest Delivery Person by City')
            df_aux = get_top_delivery_person_by_City(df1, False)
            st.dataframe(df_aux)
