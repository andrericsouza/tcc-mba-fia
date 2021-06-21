import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
import pickle 

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv("dados_baseline_media_1.csv")

# função para treinar o modelo
#def train_model():
#    data = get_data()
#    x = data.drop("MEDV",axis=1)
#    y = data["MEDV"]
#    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
#    rf_regressor.fit(x, y)
#    return rf_regressor

#importando modelo ja treinado
modelo = pickle.load(open('modelo_gb.pkl', 'rb'))

# criando um dataframe
data = get_data()

# treinando o modelo
#model = train_model()

# título
st.title("Vou bem no ENEM?")

# subtítulo
st.markdown("Este é um App para predizer se o aluno irá bem ou não no ENEM com uma solução de Machine Learning.")


# Instruções
t1 = st.subheader("Instruções:")
t2 = st.write("Preencha os dados e click em >>Realizar Predição<<")
#t3 = st.write("Abaixo do botão aparecerá o resultado")

text = '''
---
'''

st.markdown(text)


# Legenda
t4 = st.subheader("Legenda para questões de ocupação/trabalho dos pais:")

# Legenda da questao 3 (Ocupação/trabalho do pai)
t5 = st.markdown("Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultor, pescador, lenhador, seringueiro, extrativista.")
t6 = st.markdown("Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria.")
t7 = st.markdown("Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista.")
t8 = st.markdown("Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes etc.), técnico (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria.")
t9 = st.markdown("Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados.")


# verificando o dataset
##st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# atributos para serem exibidos por padrão
##defaultcols = ["NU_IDADE","TP_LINGUA","Q005","TP_COR_RACA","Q001","Q003","Q004","Q006","Q008","Q019"]

# defindo atributos a partir do multiselect
##cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# exibindo os top 10 registro do dataframe
##st.dataframe(data[cols].head(10))


#st.subheader("Distribuição...")

# definindo a faixa de valores
#faixa_valores = st.slider("Faixa de preço", float(data.MEDV.min()), 150., (10.0, 100.0))

# filtrando os dados
#dados = data[data['MEDV'].between(left=faixa_valores[0],right=faixa_valores[1])]

# plot a distribuição dos dados
#f = px.histogram(dados, x="MEDV", nbins=100, title="Distribuição de Preços")
#f.update_xaxes(title="MEDV")
#f.update_yaxes(title="Total Imóveis")
#st.plotly_chart(f)

st.sidebar.subheader("Dados do aluno")

#
# mapeando dados do usuário para cada atributo
#
#
#IDADE
#v_idade = st.sidebar.number_input("Idade:", value=18, max_value=99, min_value=0)
v_idade = 0
#
#LINGUA ESTRANGEIRA
tmp_tp_lingua = st.sidebar.selectbox("Lingua Estrangeira ?",("Inglês","Espanhol"))
# transformando o dado de entrada em valor binário
v_tp_lingua = 0 if tmp_tp_lingua == "Ingês" else 1
#valores do grafico
gl = 100 if tmp_tp_lingua == "Ingês" else 67
#
#Quantas pessoas moram na casa? - Q005
v_Q005 = st.sidebar.number_input("Quantas pessoas moram na casa?", value=1, max_value=20, min_value=0)
#valores do grafico
if v_Q005 == 1:
    g5 = 7
elif v_Q005 == 2:
    g5 = 37
elif v_Q005 == 3:
    g5 = 80
elif v_Q005 == 4:
    g5 = 100
elif v_Q005 == 5:
    g5 = 43
elif v_Q005 == 6:
    g5 = 14
elif v_Q005 == 7:
    g5 = 5
elif v_Q005 == 8:
    g5 = 2
else:
    g5 = 1
    
#
#TP_COR_RACA
tmp_TP_COR_RACA = st.sidebar.selectbox("Cor/Raça",("Não declarado","Branca","Preta","Parda","Amarela","Indígena"),1)
# transformando o dado de entrada em valor numerico
v_TP_COR_RACA1 = 1 if tmp_TP_COR_RACA == "Branca" else 0
v_TP_COR_RACA2 = 1 if tmp_TP_COR_RACA == "Preta" else 0
v_TP_COR_RACA3 = 1 if tmp_TP_COR_RACA == "Parda" else 0
v_TP_COR_RACA4 = 1 if tmp_TP_COR_RACA == "Amarela" else 0
v_TP_COR_RACA5 = 1 if tmp_TP_COR_RACA == "Indígena" else 0
#valores do grafico
if v_TP_COR_RACA3 == 1:
    gcor = 100
elif v_TP_COR_RACA1 == 1:
    gcor = 66
elif v_TP_COR_RACA2 == 1:
    gcor = 65
else:
    gcor = 10

#ESTUDO DO PAI - Q001
tmp_Q001 = st.sidebar.selectbox("Até que série seu pai, ou o homem responsável por você, estudou?",("Nunca estudou.","Não completou a 4ª/5º.","Não completou a 8ª/9º.","Não completou o Ensino Médio.","Completou o Ensino Médio.","Superior Completo","Pós-graduação.", "Não sei."),5)
v_Q001_B = 1 if tmp_Q001 == "Não completou a 4ª/5º." else 0
v_Q001_C = 1 if tmp_Q001 == "Não completou a 8ª/9º." else 0
v_Q001_D = 1 if tmp_Q001 == "Não completou o Ensino Médio." else 0
v_Q001_E = 1 if tmp_Q001 == "Completou o Ensino Médio." else 0
v_Q001_F = 1 if tmp_Q001 == "Superior Completo" else 0
v_Q001_G = 1 if tmp_Q001 == "Pós-graduação." else 0
v_Q001_H = 1 if tmp_Q001 == "Não sei." else 0
#valores do grafico
if v_Q001_B == 1:
    g1 = 100
elif v_Q001_D == 1:
    g1 = 67
else:
    g1 = 10
#
#Ocupação/trabalho do pai - Q003
tmp_Q003 = st.sidebar.selectbox("Ocupação/trabalho do pai",("Grupo 1","Grupo 2","Grupo 3","Grupo 4","Grupo 5"),4)
v_Q003_B = 1 if tmp_Q003 == "Grupo 1" else 0
v_Q003_C = 1 if tmp_Q003 == "Grupo 2" else 0
v_Q003_D = 1 if tmp_Q003 == "Grupo 3" else 0
v_Q003_E = 1 if tmp_Q003 == "Grupo 4" else 0
v_Q003_F = 1 if tmp_Q003 == "Grupo 5" else 0
#valores do grafico
if tmp_Q003 == "Grupo 5":
    g3 = 100
elif tmp_Q003 == "Grupo 2":
    g3 = 76
else:
    g3 = 10
#
#Ocupação/trabalho da mãe
tmp_Q004 = st.sidebar.selectbox("Ocupação/trabalho da mãe",("Grupo 1","Grupo 2","Grupo 3","Grupo 4","Grupo 5"),3)
v_Q004_B = 1 if tmp_Q004 == "Grupo 1" else 0
v_Q004_C = 1 if tmp_Q004 == "Grupo 2" else 0
v_Q004_D = 1 if tmp_Q004 == "Grupo 3" else 0
v_Q004_E = 1 if tmp_Q004 == "Grupo 4" else 0
v_Q004_F = 1 if tmp_Q004 == "Grupo 5" else 0
#valores do grafico
if v_Q004_B == 1:
    g4 = 100
elif v_Q004_F == 1:
    g4 = 45
elif v_Q004_C == 1:
    g4 = 23
else:
    g4 = 10
#
#Renda da família
faixa_valores = st.sidebar.slider("Renda da família", float(0), 20000., (0., 3000.0))
#faixa_valores = st.sidebar.slider("Renda da família", float(0), 20000., 3000.0)
v_Q006_B = 1 if (faixa_valores[1] > 0     and faixa_valores[1] <= 998) else 0
v_Q006_C = 1 if (faixa_valores[1] > 998   and faixa_valores[1] <= 1497 ) else 0
v_Q006_D = 1 if (faixa_valores[1] > 1497  and faixa_valores[1] <= 1996 ) else 0
v_Q006_E = 1 if (faixa_valores[1] > 1996  and faixa_valores[1] <= 2495 ) else 0
v_Q006_F = 1 if (faixa_valores[1] > 2495  and faixa_valores[1] <= 2994 ) else 0
v_Q006_G = 1 if (faixa_valores[1] > 2994  and faixa_valores[1] <= 3992 ) else 0
v_Q006_H = 1 if (faixa_valores[1] > 3992  and faixa_valores[1] <= 4990 ) else 0
v_Q006_I = 1 if (faixa_valores[1] > 4990  and faixa_valores[1] <= 5988 ) else 0
v_Q006_J = 1 if (faixa_valores[1] > 5988  and faixa_valores[1] <= 6986 ) else 0
v_Q006_K = 1 if (faixa_valores[1] > 6986  and faixa_valores[1] <= 7984 ) else 0
v_Q006_L = 1 if (faixa_valores[1] > 7984  and faixa_valores[1] <= 8982 ) else 0
v_Q006_M = 1 if (faixa_valores[1] > 8982  and faixa_valores[1] <= 9980 ) else 0
v_Q006_N = 1 if (faixa_valores[1] > 9980  and faixa_valores[1] <= 11976) else 0
v_Q006_O = 1 if (faixa_valores[1] > 11976 and faixa_valores[1] <= 14970) else 0
v_Q006_P = 1 if (faixa_valores[1] > 14970 and faixa_valores[1] <= 19960) else 0
v_Q006_Q = 1 if  faixa_valores[1] > 19960 else 0
#valores do grafico
if v_Q006_C == 1:
    g6 = 100
elif v_Q006_D == 1:
    g6 = 53
elif v_Q006_E == 1:
    g6 = 45
elif v_Q006_B == 1:
    g6 = 45
else:
    g6 = 10


#residencia tem banheiro - Q008
tmp_Q008 = st.sidebar.number_input("Quantas banheiros tem a residencia?", value=1, max_value=10, min_value=0)
v_Q008_B = 1 if tmp_Q008 == 1 else 0
v_Q008_C = 1 if tmp_Q008 == 2 else 0
v_Q008_D = 1 if tmp_Q008 == 3 else 0
v_Q008_E = 1 if tmp_Q008 >= 4 else 0
#valores do grafico
if tmp_Q008 == 1:
    g8 = 100
elif tmp_Q008 == 2:
    g8 = 20.91
elif tmp_Q008 == 3:
    g8 = 5.47
else:
    g8 = 1

#
#tem TV - Q019
tmp_Q019 = st.sidebar.number_input("Quantas TV tem a residencia?", value=1, max_value=10, min_value=0)
v_Q019_B = 1 if tmp_Q019 == 1 else 0
v_Q019_C = 1 if tmp_Q019 == 2 else 0
v_Q019_D = 1 if tmp_Q019 == 3 else 0
v_Q019_E = 1 if tmp_Q019 >= 4 else 0
#valores do grafico
if tmp_Q019 == 1:
    g19 = 100
elif tmp_Q019 == 2:
    g19 = 63
elif tmp_Q019 == 3:
    g19 = 55
else:
    g19 = 10



# inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

# verifica se o botão foi acionado
if btn_predict:
    v = [v_idade, #34, #NU_IDADE
        v_tp_lingua, #TP_LINGUA
        v_Q005, #Q005
        0, #'TP_SEXO_M'
        0, #'TP_ESTADO_CIVIL_1',
        0, #'TP_ESTADO_CIVIL_2',
        0, #'TP_ESTADO_CIVIL_3',
        0, #'TP_ESTADO_CIVIL_4',
        v_TP_COR_RACA1, #'TP_COR_RACA_1',
        v_TP_COR_RACA2, #'TP_COR_RACA_2',
        v_TP_COR_RACA3, #'TP_COR_RACA_3',
        v_TP_COR_RACA4, #'TP_COR_RACA_4',
        v_TP_COR_RACA5, #'TP_COR_RACA_5',
        0, #'TP_NACIONALIDADE_1',
        0, #'TP_NACIONALIDADE_2',
        0, #'TP_NACIONALIDADE_3',
        0, #'TP_NACIONALIDADE_4',
        0, #'TP_ST_CONCLUSAO_2',
        0, #'TP_ST_CONCLUSAO_3',
        0, #'TP_ST_CONCLUSAO_4',
        0, #'TP_ENSINO_1.0',
        0, #'TP_ENSINO_2.0',
        0, #'TP_DEPENDENCIA_ADM_ESC_1.0',
        0, #'TP_DEPENDENCIA_ADM_ESC_2.0',
        0, #'TP_DEPENDENCIA_ADM_ESC_3.0',
        0, #'TP_DEPENDENCIA_ADM_ESC_4.0',
        v_Q001_B, #'Q001_B',
        v_Q001_C, #'Q001_C',
        v_Q001_D, #'Q001_D',
        v_Q001_E, #'Q001_E',
        v_Q001_F, #'Q001_F',
        v_Q001_G, #'Q001_G',
        v_Q001_H, #'Q001_H',
        0, #'Q002_B',
        0, #'Q002_C',
        0, #'Q002_D',
        0, #'Q002_E',
        0, #'Q002_F',
        0, #'Q002_G',
        0, #'Q002_H',
        v_Q003_B, #'Q003_B',
        v_Q003_C, #'Q003_C',
        v_Q003_D, #'Q003_D',
        v_Q003_E, #'Q003_E',
        v_Q003_F, #'Q003_F',
        v_Q004_B, #'Q004_B',
        v_Q004_C, #'Q004_C',
        v_Q004_D, #'Q004_D',
        v_Q004_E, #'Q004_E',
        v_Q004_F, #'Q004_F',
        v_Q006_B, #'Q006_B',
        v_Q006_C, #'Q006_C',
        v_Q006_D, #'Q006_D',
        v_Q006_E, #'Q006_E',
        v_Q006_F, #'Q006_F',
        v_Q006_G, #'Q006_G',
        v_Q006_H, #'Q006_H',
        v_Q006_I, #'Q006_I',
        v_Q006_J, #'Q006_J',
        v_Q006_K, #'Q006_K',
        v_Q006_L, #'Q006_L',
        v_Q006_M, #'Q006_M',
        v_Q006_N, #'Q006_N',
        v_Q006_O, #'Q006_O',
        v_Q006_P, #'Q006_P',
        v_Q006_Q, #'Q006_Q',
        0, #'Q007_B',
        0, #'Q007_C',
        0, #'Q007_D',
        v_Q008_B, #'Q008_B',
        v_Q008_C, #'Q008_C',
        v_Q008_D, #'Q008_D',
        v_Q008_E, #'Q008_E',
        0, #'Q009_B',
        0, #'Q009_C',
        0, #'Q009_D',
        0, #'Q009_E',
        0, #'Q010_B',
        0, #'Q010_C',
        0, #'Q010_D',
        0, #'Q010_E',
        0, #'Q011_B',
        0, #'Q011_C',
        0, #'Q011_D',
        0, #'Q011_E',
        0, #'Q012_B',
        0, #'Q012_C',
        0, #'Q012_D',
        0, #'Q012_E',
        0, #'Q013_B',
        0, #'Q013_C',
        0, #'Q013_D',
        0, #'Q013_E',
        0, #'Q014_B',
        0, #'Q014_C',
        0, #'Q014_D',
        0, #'Q014_E',
        0, #'Q015_B',
        0, #'Q015_C',
        0, #'Q015_D',
        0, #'Q015_E',
        0, #'Q016_B',
        0, #'Q016_C',
        0, #'Q016_D',
        0, #'Q016_E',
        0, #'Q017_B',
        0, #'Q017_C',
        0, #'Q017_D',
        0, #'Q017_E',
        0, #'Q018_B',
        v_Q019_B, #'Q019_B',
        v_Q019_C, #'Q019_C',
        v_Q019_D, #'Q019_D',
        v_Q019_E, #'Q019_E',
        0, #'Q020_B',
        0, #'Q021_B',
        0, #'Q022_B',
        0, #'Q022_C',
        0, #'Q022_D',
        0, #'Q022_E',
        0, #'Q023_B',
        0, #'Q024_B',
        0, #'Q024_C',
        0, #'Q024_D',
        0, #'Q024_E',
        0, #'Q025_B',
        0, #'LE_NO_MUNICIPIO_RESIDENCIA',
        0, #'LE_NO_MUNICIPIO_NASCIMENTO',
        0, #'LE_SG_UF_NASCIMENTO',
        0#, #'LE_NO_MUNICIPIO_ESC',
        #0, #'LE_SG_UF_PROVA'
         ]
    #result = modelo.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    result = modelo.predict([v])
    r = result[0] #st.write(result[0])
    if r == 1:
        v_resultado = "Aprovado!" #if r == 1  else "erro no calculo1"
        g_cor = 'g'
    else:
        v_resultado = "Reprovado!" # if r == 0 else "erro no calculo2"
        g_cor = 'r'
    #st.sidebar.subheader("O Aluno foi:")
    st.subheader("O Aluno foi:")
    #result = "US $ "+str(round(result[0]*10,2))
    #v_resultado = str(result)
    #st.sidebar.write(v_resultado)
    st.write(v_resultado)

    # plot a distribuição dos dados
    #f = px.histogram(dados, x="MEDV", nbins=100, title="Distribuição de Preços")
    #f.update_xaxes(title="MEDV")
    #f.update_yaxes(title="Total Imóveis")
    #st.plotly_chart(f)

    caracteristicas = ["TRABALHO_PAI", "TRABALHO_MÃE", "ESTUDO_PAI", "BANHEIRO", "TV", "MORADORES", "RAÇA", "RENDA", "IDIOMA"]
    valor = [g3, g4, g1, g8, g19, g5, gcor, g6 , gl, g3]
    #valor = [45, 53, 15, 61, 57, 45, 30, 30 , 90, 45]
     
    # Initialise the spider plot by setting figure size and polar projection
    fig = plt.figure(figsize=(10, 6))
    plt.subplot(polar=True)
     
    theta = np.linspace(0, 2 * np.pi, len(valor))
     
    # Arrange the grid into number of sales equal parts in degrees
    lines, labels = plt.thetagrids(range(0, 360, int(360/len(caracteristicas))), (caracteristicas))
     
    # Plot actual sales graph
    plt.plot(theta, valor)
    plt.fill(theta, valor, g_cor, alpha=0.2)
     
    # Plot expected sales graph
    #plt.plot(theta, expected)
     
    # Add legend and title for the plot
    #plt.legend(labels=('Actual', 'Expected'), loc=1)
    plt.title("Vou bem no ENEM? \n")
     
    # Dsiplay the plot on the screen
    #plt.show()
    #st.plotly_chart(fig)
    t1.empty()
    #t2.write("")
    #t3.empty()
    t4.empty()
    t5.empty()
    t6.empty()
    t7.empty()
    t8.empty()
    t9.empty()
    st.write(fig)
    #st.write(valor)