

import numpy as np
import pandas as pd

def clean_price(val):
    if type(val) == str:
        if val.count('.') > 1:
            tam_kisim, ondalik = val.rsplit('.', 1)
            tam_kisim = tam_kisim.replace('.', '')
            return float(tam_kisim + "." + ondalik)
        return float(val)
    return float(val)

def parse_date(date_str):
    # Türkçe ayları ve verideki 'Ţubat' hatasını sayılara (01, 02) çevirir.
    aylar = {'Ocak': '01', 'Şubat': '02', 'Ţubat': '02', 'Mart': '03', 'Nisan': '04','Mayýs':'05',
             'Mayıs': '05', 'Haziran': '06', 'Temmuz': '07', 'Ağustos': '08','Ađustos':'08',
             'Eylül': '09', 'Ekim': '10', 'Kasım': '11','Kasým':'11', 'Aralık': '12','Aralýk':'12'}
    for ay, sayi in aylar.items():
        date_str = date_str.replace(ay, sayi)
    return date_str

def load_and_preprocess_data(filepath):
    df=pd.read_csv(filepath,sep=';')
    df.columns=df.columns.str.strip()

    df['Acilis'] = df['Acilis'].apply(clean_price)
    df['Kapanis'] = df['Kapanis'].apply(clean_price)

    df['Tarih']=df['Tarih'].apply(parse_date)
    df['Tarih']=pd.to_datetime(df['Tarih'] ,format='%d %m %Y')

    df=df.sort_values('Tarih').ffill()
    df['Prev_Acilis'] = df['Acilis'].shift(1)
    df['Prev_Kapanis'] = df['Kapanis'].shift(1)
    

    return df.dropna().reset_index(drop=True)

def train_model(X_train,y_train,lr=0.1,epochs=1000):
    n_samples,n_features =X_train.shape
    w,b=np.zeros(n_features),0.0

    for _ in range(epochs):
        y_pred=np.dot(X_train,w)+b
        error= y_pred-y_train

        dw=(1/n_samples)*np.dot(X_train.T,error)
        db=(1/n_samples)*np.sum(error)
        w -= lr * dw
        b -= lr * db

    return w,b

def evaluate_model(w,b,X,y):
    y_pred=np.dot(X,w)+b

    mse=np.mean((y-y_pred)**2)
    r2=1-(np.sum((y-y_pred)**2)/np.sum((y-np.mean(y))**2))

    return mse,r2
