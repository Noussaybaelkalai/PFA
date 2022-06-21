from streamlit_option_menu import option_menu
from PIL import Image
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
from forex_python.converter import CurrencyRates,CurrencyCodes
from forex_python.bitcoin import BtcConverter
import pandas as pd
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance  as yf
import numpy as np
import math
import plotly.express as px
import pickle
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

conn = sqlite3.connect('data.db')
c = conn.cursor()

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def create_user_table():
    c.execute('CREATE TABLE IF NOT EXISTS usertable(username TEXT ,password TEXT) ')

def add_user_data(username,password):
    c.execute('INSERT INTO usertable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM usertable WHERE username=? AND password=?',(username,password))
    data = c.fetchall()
    return data
def view_all_user():
    c.execute('SELECT * FROM usertable')
    data = c.fetchall()
    return data


logo = Image.open(r'C:\Users\HP\Downloads\BTC.png')
st.sidebar.image(logo,width=170)

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ['Dashboard','Convertir',"A propos", "Arbitrage triangulaire", "Signup"],
        icons=['columns', 'arrow-left-right','house-door',"recycle", 'person-plus-fill'], menu_icon="cast", default_index=1)




if selected == "Arbitrage triangulaire":

    # --- USER AUTHENTICATION ---
    names = ["elkalai", "Rotbi"]
    usernames = ["Noussayba", "Deyae"]

    # load hashed passwords
    file_path = Path(__file__).parent / "hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "BTC_dashboard", "abcdef",
                                        cookie_expiry_days=30)

    name, authentication_status, username = authenticator.login("Log In", "main")

    if authentication_status == False:
        st.error("Nom d'utilisateur /Mot de pass est incorrect")

    if authentication_status == None:
        st.warning("Veuillez entrer votre nom d'utilisateur et mot de pass")
        st.warning("Sign Up Now !")

    if authentication_status:
        st.success("Logged in as {}".format(username))
        st.title("Etape 1:")
        menu3 = ["USD", "EUR", "GBP", "CAD", "JPY"]
        menu4 = ["EUR", "USD", "GBP", "CAD", "JPY"]
        c = CurrencyRates()
        amount = st.number_input('Entrer le montant :')
        from_currency = st.selectbox('De la monnaie :',menu4)
        to_currency = st.selectbox('Vers la monnaie:',menu3)
        result = c.convert(from_currency, to_currency, amount)
        st.title(result)

        st.title("Etape 2:")

        c1 = CurrencyRates()
        menu5 = ["JPY","EUR", "USD", "GBP", "CAD"]
        to_currency1 = st.selectbox('Vers la monnaie secondaire:',menu5)
        result1 = c1.convert(to_currency, to_currency1, result)
        st.title(result1)

        st.title("Etape 3:")
        c2 = CurrencyRates()
        to_currency2 = st.selectbox('Vers la monnaie tertière:',menu4)
        result2 = c2.convert(to_currency1, to_currency2, result1)
        st.title(result2)

        authenticator.logout("Log Out", "sidebar")

if selected == "Signup":
    st.title("Authentification")
    st.subheader("Créer un nouveau compte")
    new_user = st.text_input("Nom de l'utilisateur")
    new_password = st.text_input("Mot de pass", type='password')
    if st.button("Sign Up"):
        create_user_table()
        add_user_data(new_user, new_password)
        st.success("Vous avez créer un compte avec succès")
        st.info("Retourner au menu principale pour se connecter ")



if selected =="Dashboard":
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Market cap",
                value=764.900224789,
                delta=-0.637)
    kpi2.metric(label="Fully diluted "
                      "Market cap",
                value=899.700524889,
                delta=-0.637)
    kpi3.metric(label="Volume(24H)",
                value=17.6409002,
                delta=5.12)

    data = pd.read_csv('C:/Users/HP/Downloads/BTC.csv', header=0, index_col='Date', parse_dates=True)

    # create a new dataframe with only close column
    data = data.filter(['Close'])

    # convert dataframe to a numpy array
    dataset = data.values

    # get the number of rows to train the model on
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # load model
    model = load_model('keras_model.h5')

    # Create the testing dataset
    test_data = scaled_data[training_data_len - 60:, :]
    # Create the dataset x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the model predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions



    chart1, chart2 = st.columns(2)
    # Show the valid and predictions prices
    chart1.write(valid['Predictions'])

    chart2.write("Pour voir le modèle historique ou bien la prédiction" 
                  "  dans l'année prochaine de prix de BTC "
                  " cliquez sur le boutou ci-dessous : ")

    menu2 = ["Model historique", "Prédiction"]
    choice2 = chart2.selectbox("View Charts", menu2)

    if choice2 == "Model historique":
        fig1 = px.line(train['Close'], title='Model historique')
        fig1.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1TD", step="day", stepmode="todate"),
                    dict(count=1, label="1MTD", step="month", stepmode="todate"),
                    dict(count=6, label="6MTD", step="month", stepmode="todate"),
                    dict(count=1, label="1YTD", step="year", stepmode="todate"),
                    dict(count=2, label="2YTD", step="year", stepmode="todate"),
                    dict(count=3, label="3YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
        )
        fig1.show()


    elif choice2 == "Prédiction":
        fig2 = px.line(valid['Predictions'], title='Prediction')
        fig2.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1TD", step="day", stepmode="todate"),
                    dict(count=1, label="1MTD", step="month", stepmode="todate"),
                    dict(count=2, label="2MTD", step="month", stepmode="todate"),
                    dict(count=6, label="6MTD", step="month", stepmode="todate"),
                    dict(count=1, label="1YTD", step="year", stepmode="todate"),
                    dict(step="all")
                ])
            )
        )
        fig2.show()



if selected =="Convertir":
    st.title("Prix de BTC en USD:")
    bitcoin = BtcConverter()
    price = bitcoin.get_latest_price("USD")
    st.header(price)
    c = CurrencyRates()
    amount3 = st.number_input('Entrer le montant :')
    menu3 = ["USD", "EUR","GBP","CAD","JPY"]
    menu4 = ["EUR","USD", "GBP", "CAD", "JPY"]
    from_currency3 = st.selectbox('De la monnaie :',menu3)
    to_currency3 = st.selectbox('Vers la monnaie:',menu4)
    res = c.convert(from_currency3, to_currency3, amount3)
    st.header(res)

if selected =="A propos":
    st.title("Bitcoin ")
    st.text("Bitcoin est une technologie pair à pair fonctionnant sans autorité centrale. ")
    st.text("Bitcoin est libre et ouvert. Sa conception est publique, personne ne possède ni ")
    st.text("ne contrôle Bitcoin et tous peuvent s'y joindre.Grâce à plusieurs de ses propriétés  ")
    st.text("uniques,Bitcoin rend possible des usages prometteurs qui ne pourraient pas être  ")
    st.text("couverts par les systèmes de paiement précédents.")
    img = Image.open(r'C:\Users\HP\Downloads\btc-c.png')
    st.image(img, width=170)


