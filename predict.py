import numpy as np
from model.functions import load_and_preprocess_data, train_model, evaluate_model

def main():
    df=load_and_preprocess_data('dataset/gram_gold_10yrs.csv')
    X=df[['Prev_Acilis','Prev_Kapanis']].values
    y = df['Kapanis'].values

    n=len(df)
    val_idx=int(n*0.70)
    test_idx=int(n*0.85)

    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]

    X_mean,X_std =np.mean(X_train,axis=0),np.std(X_train,axis=0)
    X_train_scaled=(X_train-X_mean)/X_std
    X_val_scaled=(X_val-X_mean)/X_std
    X_test_scaled=(X_test-X_mean)/X_std


    w,b=train_model(X_train_scaled,y_train,lr=0.1,epochs=1000)
    val_mse, val_r2 = evaluate_model(w, b, X_val_scaled, y_val)
    test_mse, test_r2 = evaluate_model(w, b, X_test_scaled, y_test)

    print(f"Test Verisi MSE (Hata)  : {test_mse:.2f}")
    print(f"Test Verisi R2 (Başarı) : %{test_r2 * 100:.2f}")

    last_row = df.iloc[-1]
    next_day_features = np.array([last_row['Acilis'], last_row['Kapanis']])
    next_day_scaled = (next_day_features - X_mean) / X_std

    predicted_price = np.dot(next_day_scaled, w) + b

    print(f"Verisetindeki Son Tarih : {next_day_features}")
    print(f"Bilinen Son Kapanış     : {last_row['Kapanis']:.2f} TL")
    print(f"---------------------------------------")
    print(f"TAHMİN EDİLEN YENİ FİYAT: {predicted_price:.2f} TL")

if __name__ == "__main__":
    main()




    