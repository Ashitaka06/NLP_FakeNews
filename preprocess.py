# Chargement des données d'entraînement à partir de train_bodies.csv et train_stances.csv. Prétraitement des données (par exemple, nettoyage du texte, tokenisation).
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
import os
from tqdm import tqdm

def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        print("Punkt est déjà présent dans l'environnement.")
    except LookupError:
        print("Téléchargement de Punkt...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
        print("Les stopwords sont déjà présents dans l'environnement.")
    except LookupError:
        print("Téléchargement des stopwords...")
        nltk.download('stopwords')

def clean_text(text):
    # Convertir la donnée en chaîne de caractères si ce n'est pas déjà le cas
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords.words('english')]
    return ' '.join(tokens)

def map_label(label):
    truth_labels = ['mostly-true', 'half-true']
    if label in truth_labels:
        return 'Info'
    else:
        return 'Intox'
    
from translatepy import Translator

def translate_text(text):
    translator = Translator()
    try:
        # Traduit le texte en français
        translation = translator.translate(text, "French")
        # Retourne uniquement le résultat de la traduction
        return translation.result
    except Exception as e:
        print(f"Erreur lors de la traduction : {e}")
        return text

def load_and_merge_tsv_files():
    print("Lecture des fichiers TSV...")
    test = pd.read_csv('test.tsv', sep='\t', header=None, usecols=[1, 2])
    train = pd.read_csv('train.tsv', sep='\t', header=None, usecols=[1, 2])
    valid = pd.read_csv('valid.tsv', sep='\t', header=None, usecols=[1, 2])

    print("Fusion des fichiers...")
    merged_data = pd.concat([test, train, valid], ignore_index=True)

    return merged_data

def preprocess_merged_data(data):
    print("Nettoyage du texte et mise à jour des labels...")
    
    tqdm.pandas(desc="Mise à jour des labels")
    data[1] = data[1].apply(map_label) 
    
    tqdm.pandas(desc="Nettoyage du texte")
    data[2] = data[2].apply(clean_text)

    # Renommer les colonnes pour plus de clarté
    data.columns = ['label', 'affirmation']
    
    return data

def preprocess_merged_translated_data(data):
    print("Nettoyage du texte, mise à jour des labels et traduction...")
    
    tqdm.pandas(desc="Mise à jour des labels")
    data[1] = data[1].progress_apply(map_label)
    
    tqdm.pandas(desc="Nettoyage du texte")
    data[2] = data[2].progress_apply(clean_text)
    
    tqdm.pandas(desc="Traduction du texte")
    data[2] = data[2].progress_apply(translate_text)

    # Renommer les colonnes pour plus de clarté
    data.columns = ['label', 'affirmation']
    
    return data

def load_preprocessed_data():
    if os.path.exists('preprocessed_data_merged.csv'):
        print("Chargement des données prétraitées...")
        return pd.read_csv('preprocessed_data_merged.csv')
    else:
        print("Prétraitement des données...")
        data = load_and_merge_tsv_files()
        data = preprocess_merged_data(data)
        # Sauvegarder pour une utilisation future
        data.to_csv('preprocessed_data_merged.csv', index=False)
        return data
    
def load_preprocessed_translated_data():
    if os.path.exists('preprocessed_data_merged_fr.csv'):
        print("Chargement des données prétraitées avec traduction...")
        return pd.read_csv('preprocessed_data_merged_fr.csv')
    else:
        print("Prétraitement des données et traduction...")
        data = load_and_merge_tsv_files()
        data = preprocess_merged_translated_data(data)
        # Sauvegarder pour une utilisation future
        data.to_csv('preprocessed_data_merged_fr.csv', index=False)
        return data

# Téléchargez les données NLTK nécessaires
download_nltk_data()

# Charger et prétraiter les données
data = load_preprocessed_data()
print(data.head())

data_fr = load_preprocessed_translated_data()
print(data_fr.head())