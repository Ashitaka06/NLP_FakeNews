import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import load

# Charger le modèle et le vectoriseur
TAILLE_ENTREE = 1000
TAILLE_CACHEE = 200
NB_CLASSES = 2
MODELE_CHEMIN = 'model.ckpt'
VECTEUR_CHEMIN = 'tfidf_vectorizer.joblib'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Définition du modèle
class ModeleDetectionPosition(nn.Module):
    def __init__(self, taille_entree, taille_cachee, nb_classes):
        super(ModeleDetectionPosition, self).__init__()
        self.fc1 = nn.Linear(taille_entree, taille_cachee)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(taille_cachee, nb_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

modele = ModeleDetectionPosition(TAILLE_ENTREE, TAILLE_CACHEE, NB_CLASSES)
modele.load_state_dict(torch.load(MODELE_CHEMIN, map_location=device))
modele.to(device)
modele.eval()

tfidf_vectorizer = load(VECTEUR_CHEMIN)

# Fonction pour prédire
def predire(texte):
    caracteristiques = tfidf_vectorizer.transform([texte]).toarray()
    caracteristiques_tensor = torch.tensor(caracteristiques, dtype=torch.float32).to(device)
    with torch.no_grad():
        sorties = modele(caracteristiques_tensor)
        probas = torch.nn.functional.softmax(sorties, dim=1)
        classe = probas.argmax(dim=1).item()
        proba = probas[0, classe].item()
    return classe, proba

# Charger le jeu de données de test
df = pd.read_csv('amaury_test.csv')
predictions = []
probas = []

# Faire des prédictions sur chaque affirmation
for texte in df['affirmation']:
    classe_predite, proba = predire(texte)
    predictions.append("Info" if classe_predite == 0 else "Intox")
    probas.append(proba)

# Ajouter les prédictions au DataFrame
df['prediction'] = predictions
df['proba'] = probas

# Calculer les métriques de performance
verite = df['label']
pred = df['prediction']
print("Précision:", accuracy_score(verite, pred))
print("Rappel:", recall_score(verite, pred, pos_label="Intox"))
print("Score F1:", f1_score(verite, pred, pos_label="Intox"))
print("Matrice de confusion:\n", confusion_matrix(verite, pred))