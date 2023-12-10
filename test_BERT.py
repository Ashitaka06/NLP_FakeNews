import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Paramètres du modèle
MODELE_BERT_CHEMIN = 'model.pt'  # Chemin de votre fichier .pt

# Charger le modèle et le tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modele_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
etat_modele = torch.load(MODELE_BERT_CHEMIN, map_location=device)
modele_bert.load_state_dict(etat_modele)
modele_bert.to(device)
modele_bert.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predire_bert(texte):
    inputs = tokenizer(texte, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        sorties = modele_bert(**inputs)
    probas = torch.nn.functional.softmax(sorties.logits, dim=1)
    classe = probas.argmax(dim=1).item()
    return classe

# Charger les données de test
df = pd.read_csv('amaury_test.csv')
y_true = df['label'].apply(lambda x: 0 if x == 'Info' else 1).tolist()
y_pred = [predire_bert(texte) for texte in df['affirmation']]

# Calculer les métriques
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

# Afficher les résultats
print(f'Précision : {accuracy:.4f}')
print(f'Rappel : {recall:.4f}')
print(f'Score F1 : {f1:.4f}')
print('Matrice de confusion :')
print(conf_matrix)
