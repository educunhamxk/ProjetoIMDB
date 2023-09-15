import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Stopwords em português
stop_words_portuguese = stopwords.words('portuguese')

#Configuração de página extendida
st.set_page_config(layout="wide") 

#Imagem de fundo e título
st.markdown("<h1 style='text-align: center; color: black;'>Posso te recomendar algo para assistir com base no seu gosto?</h1>", unsafe_allow_html=True)
st.image('filmes.jpg', use_column_width=True)

#Carregamento do dataframe
@st.cache_resource
def carregar_dados():
    return pd.read_excel("df_final_imdb.xlsx")

df = carregar_dados()

#filtrar títulos que possuem imagens
titulos_com_imagens = []

for _, row in df.iterrows():
    title_pt = row['title_pt']
    title_en = row['title_en']

    file_path_pt = title_pt + ".jpg"
    file_path_en = title_en + ".jpg"
    
    if os.path.exists(file_path_pt) or os.path.exists(file_path_en):
        titulos_com_imagens.append(row)

#Geração do DF
df = pd.DataFrame(titulos_com_imagens)
df['sinopse_no_stopwords'].fillna('', inplace=True)

#Parametrização para funcionamento das listas nos streamlit
if 'index_lista' not in st.session_state:
    st.session_state.index_lista = 0

if 'titulos_selecionados' not in st.session_state:
    st.session_state.titulos_selecionados = []

if 'listas_titulos' not in st.session_state:
    st.session_state.listas_titulos = [df.sample(10)['title_en'].tolist() for _ in range(10)]

#Enquanto tiver gerado menos de 10 listas de filmes...
if st.session_state.index_lista < 10:
    st.write(f"Escolha o seu filme favorito da lista {st.session_state.index_lista + 1}:")

    #Dividindo a página em 10 espaços
    cols = st.columns(10)

    #Para cada titulo será exibida a imagem do filme e as informações do título serão capturadas
    for idx, title in enumerate(st.session_state.listas_titulos[st.session_state.index_lista]):
        file_path_pt = title + ".jpg"
        
        title_en_rows = df[df['title_pt'] == title]['title_en']
        file_path_en = title_en_rows.values[0] + ".jpg" if not title_en_rows.empty else None
        
        #Tratamento para tentar buscar as imagens tanto com o título em inglês quanto em português
        if os.path.exists(file_path_pt):
            image_path = file_path_pt
        elif file_path_en and os.path.exists(file_path_en):
            image_path = file_path_en
        else:
            image_path = None

        #Para cada espaço é exibida a imagem e o botão de seleção do filme
        with cols[idx]:
            if image_path:
                st.image(image_path, width=100) 
            if st.button(title):
                st.session_state.titulos_selecionados.append(title)
                st.session_state.index_lista += 1

#Se passar de 10 títulos selecionados então serão geradas as recomendações
else:
    with st.spinner('Aguarde enquanto buscamos as melhores recomendações para você...'):
        #Adicionar a nova coluna para marcação dos filmes que foram escolhidos
        # df['escolhido'] = 0
        # df.loc[df['title_pt'].isin(st.session_state.titulos_selecionados), 'escolhido'] = 1

        # #Marcação dos filmes que não foram escolhidos
        # if st.session_state.index_lista < len(st.session_state.listas_titulos):
        #     df.loc[df['title_pt'].isin(st.session_state.listas_titulos[st.session_state.index_lista]), 'escolhido'] = -1
        
        #Média e Desvio Padrão de rating dos filmes selecionados
        filmes_selecionados = df[df['title_pt'].isin(st.session_state.titulos_selecionados)]
        media_rating = filmes_selecionados['rating'].mean()
        desvio_padrao = filmes_selecionados['rating'].std()

        #Filtrar df se o desvio padrão for menor ou igual a 0,5 para capturar ratings próximos a média
        if desvio_padrao <= 0.5:
            df = df[df['rating'].between(media_rating - 0.5, media_rating + 0.5)]

        #Filtrar por gêneros se o usuário escolheu filmes de até dois gêneros
        generos_selecionados = filmes_selecionados['genre'].unique()
        if len(generos_selecionados) == 2:
            df = df[df['genre'].isin(generos_selecionados)]

        #Marcação dos filmes escolhidos na base
        df['escolhido'] = 0
        df.loc[df['title_pt'].isin(st.session_state.titulos_selecionados), 'escolhido'] = 1

        # Checando se o índice é válido para marcação dos filmes que não foram escolhidos
        if 0 <= st.session_state.index_lista < len(st.session_state.listas_titulos):
            filmes_nao_escolhidos = st.session_state.listas_titulos[st.session_state.index_lista]
            df.loc[df['title_pt'].isin(filmes_nao_escolhidos), 'escolhido'] = -1

        #df.loc[df['title_pt'].isin(st.session_state.listas_titulos[st.session_state.index_lista]), 'escolhido'] = -1

        #Preparando os dados para clusterização
        vetorizador = TfidfVectorizer(stop_words=stop_words_portuguese)
        df['year'] = df['year'].astype(float)
        df['rating'] = df['rating'].astype(float)
        genre_dummies = pd.get_dummies(df['genre'], prefix='genre')
        df = pd.concat([df, genre_dummies], axis=1)
        df.drop('genre', axis=1, inplace=True)

        colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        colunas_categoricas = df.select_dtypes(include=['object']).columns.tolist()

        #Removendo a coluna 'sinopse_no_stopwords' das categóricas, pois ela será tratada separadamente.
        colunas_categoricas.remove('sinopse_no_stopwords')
        dummies = pd.get_dummies(df[colunas_categoricas], drop_first=True)

        #Transformando a coluna 'sinopse_no_stopwords' usando TF-IDF
        sinopse_tfidf = vetorizador.fit_transform(df['sinopse_no_stopwords'].fillna(''))

        #Convertendo a matriz para um dataframe
        dados_cluster = pd.DataFrame(sinopse_tfidf.toarray())
        scaler = StandardScaler()
        dados_cluster = scaler.fit_transform(dados_cluster)

        #Clusterização
        kmeans = KMeans(n_clusters=10) 
        df['cluster'] = kmeans.fit_predict(dados_cluster)

        #Descobrir qual cluster tem a maioria dos filmes escolhidos pelo usuário
        cluster_escolhido = df[df['escolhido'] == 1]['cluster'].mode()[0]

        #Selecionar filmes desse cluster
        cluster_df = df[df['cluster'] == cluster_escolhido]

        #Calculando similaridade do cosseno para recomendação
        matriz_tfidf = vetorizador.transform(cluster_df['sinopse_no_stopwords'].fillna(''))
        filmes_gostados_tfidf = vetorizador.transform(df[df['escolhido'] == 1]['sinopse_no_stopwords'])
        similaridade_cosseno = cosine_similarity(filmes_gostados_tfidf, matriz_tfidf)
        indices_recomendados = similaridade_cosseno.argsort(axis=1)[:,-10:]
        filmes_recomendados = cluster_df.iloc[indices_recomendados.flatten()].drop_duplicates().head(10)

        #Certificando-se de que os filmes escolhidos não estão nas recomendações
        filmes_recomendados = filmes_recomendados[~filmes_recomendados['title_pt'].isin(st.session_state.titulos_selecionados)]

        #Exibindo as recomendações
        st.title('Recomendações:')
        recom_cols = st.columns(10)
        idx = 0

        for _, filme in filmes_recomendados.iterrows():
            title_pt = filme['title_pt']
            title_en = filme['title_en']
            
            file_path_pt = title_pt + ".jpg"
            file_path_en = title_en + ".jpg"
            
            if os.path.exists(file_path_pt):
                image_path = file_path_pt
            elif os.path.exists(file_path_en):
                image_path = file_path_en
            else:
                image_path = None

            with recom_cols[idx]:
                if image_path:
                    st.image(image_path, width=100)
                st.write(title_pt)
                idx += 1
        
        