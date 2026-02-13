""" Fonction de preprocessing des données, à appliquer après le split en train/val/test pour avoir
    un dataset correct . """

from sklearn.preprocessing import LabelEncoder

def clean_dataset(df , df_train, df_val,  df_test):
    """
    Applique les choix VALIDÉS lors de l'EDA.
    Renvoie les datasets nettoyés, avec leur splits initiaux.
    """

    # supprimer les styles avec moins de 400 oeuvres
    style_counts = df["style_name"].value_counts()
    valid_styles = style_counts[style_counts >= 400].index
    df = df[df["style_name"].isin(valid_styles)]
    df_train = df_train[df_train["style_name"].isin(valid_styles)]
    df_val = df_val[df_val["style_name"].isin(valid_styles)]
    df_test = df_test[df_test["style_name"].isin(valid_styles)]

    le = LabelEncoder()

    df_train["style_encoded"] = le.fit_transform(df_train["style"])
    df_val["style_encoded"]   = le.transform(df_val["style"])
    df_test["style_encoded"]  = le.transform(df_test["style"])
    df["style_encoded"]      = le.transform(df["style"])
    
    # autres transfo si on a envie de faire du nettoyage

    return df, df_train, df_val, df_test

def get_split_artist(df):
    #TODO 
    # trouver une bonne méthode pour split par artiste
    # return df_train, df_val, df_test
    pass

def data_augmentation(df):
    #TODO
    # Pour faire la data augmentation si besoin 

    pass
