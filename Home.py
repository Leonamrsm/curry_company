import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"

)

# ==============================================================================
# Sidebar
# ==============================================================================

image_path = 'logo.png'
Image.open(image_path)
st.sidebar.image(image_path, width=270)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""___""")

st.sidebar.markdown('### Powered by Leonam Rezende')

st.write('# Cury Company Growth Dashboard')

st.markdown(
    """
    Growth Dashboard foi contruído para acompanhar as métricas de crescimento de Entregadores e Restaurantes.
    ### Como utilizar o Growth Dashboard?
    - Visão Empresa:
        - Visaão Gerencial: Métricas Gerais de comportamento.
        - Visão Tática: Analise de Indicadores semanais.
        - Visão Geográfica: Insights de geolocalização.
    - Visão Entegrador:
        - Acompanhamento dos indicadores semanais de crescimento.
    - Visão Restaurantes:
        - Acompanhamento dos indicadores semanais de crescimento.
    
    #### Ask for Help
        - leonamrsm@gmail.com
    
    """
)