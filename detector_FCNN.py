import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from joblib import load

# Paramètres du modèle
TAILLE_ENTREE = 1000
TAILLE_CACHEE = 200
NB_CLASSES = 2
MODELE_CHEMIN = 'best_model.ckpt'
VECTEUR_CHEMIN = 'tfidf_vectorizer.joblib'

# Définition du modèle
class ModeleDetectionPosition(nn.Module):
    def __init__(self, taille_entree, taille_cachee, nb_classes):
        super(ModeleDetectionPosition, self).__init__()
        self.fc1 = nn.Linear(taille_entree, taille_cachee)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(taille_cachee, nb_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Charger le modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modele = ModeleDetectionPosition(TAILLE_ENTREE, TAILLE_CACHEE, NB_CLASSES)
modele.load_state_dict(torch.load(MODELE_CHEMIN, map_location=device))
modele.to(device)
modele.eval()

# Charger le vectoriseur TF-IDF
if os.path.exists(VECTEUR_CHEMIN):
    tfidf_vectorizer = load(VECTEUR_CHEMIN)
else:
    raise Exception("Fichier vectoriseur TF-IDF introuvable.")

def predire(texte):
    caracteristiques = tfidf_vectorizer.transform([texte]).toarray()
    caracteristiques_tensor = torch.tensor(caracteristiques, dtype=torch.float32).to(device)
    with torch.no_grad():
        sorties = modele(caracteristiques_tensor)
        probas = torch.nn.functional.softmax(sorties, dim=1)
        classe = probas.argmax(dim=1).item()
        proba = probas[0, classe].item()
    return classe, proba

# Interface utilisateur
if __name__ == "__main__":
    texte_utilisateur = input("Entrez une phrase pour la détection : ")
    classe, proba = predire(texte_utilisateur)
    resultat = "Info" if classe == 0 else "Intox"  # Adaptez selon le codage de vos classes
    print(f"Résultat : {resultat}, Probabilité : {proba:.4f}")
