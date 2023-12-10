import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from joblib import dump
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Chemins des fichiers et paramètres
DATA = 'preprocessed_data_merged.csv'
BEST_METRICS = 'best_metrics.json'
BEST_MODEL = 'best_model.ckpt'
LAST_MODEL = 'last_model.ckpt'
VECTORIZER = 'tfidf_vectorizer.joblib'
TAILLE_ENTREE = 1000
TAILLE_CACHEE = 200 # Augmenté pour une capacité de modélisation plus élevée
NB_CLASSES = 2
TAILLE_LOT = 64 # Augmenté pour utiliser plus efficacement le GPU
NB_EPOCHS = 42 # Augmenté pour permettre un apprentissage plus approfondi
TAUX_APPRENTISSAGE = 0.001

def charger_et_pretraiter_donnees(chemin_fichier):
    donnees = pd.read_csv(chemin_fichier)
    vectoriseur = TfidfVectorizer(max_features=1000)
    caracteristiques = vectoriseur.fit_transform(donnees['affirmation']).toarray()
    encodeur_label = LabelEncoder()
    etiquettes = encodeur_label.fit_transform(donnees['label'])
    dump(vectoriseur, VECTORIZER)
    return caracteristiques, etiquettes

def creer_chargeur_donnees(caracteristiques, etiquettes, taille_lot=TAILLE_LOT):
    caracteristiques_tensor = torch.tensor(caracteristiques, dtype=torch.float32)
    etiquettes_tensor = torch.tensor(etiquettes, dtype=torch.long)
    ensemble_donnees = TensorDataset(caracteristiques_tensor, etiquettes_tensor)
    return DataLoader(ensemble_donnees, batch_size=taille_lot, shuffle=True)

def evaluer_modele(modele, chargeur_donnees, critere):
    modele.eval()
    perte_totale, precision_totale, f1_totale, rappel_totale = 0, 0, 0, 0
    with torch.no_grad():
        for caracteristiques, etiquettes in chargeur_donnees:
            caracteristiques, etiquettes = caracteristiques.to(device), etiquettes.to(device)
            sorties = modele(caracteristiques)
            perte = critere(sorties, etiquettes)
            perte_totale += perte.item()
            _, predit = torch.max(sorties.data, 1)
            precision_totale += (predit == etiquettes).sum().item()
            f1_totale += f1_score(etiquettes.cpu().numpy(), predit.cpu().numpy(), average='weighted')
            rappel_totale += recall_score(etiquettes.cpu().numpy(), predit.cpu().numpy(), average='weighted', zero_division=0)
    return perte_totale / len(chargeur_donnees), precision_totale / len(chargeur_donnees.dataset), f1_totale / len(chargeur_donnees), rappel_totale / len(chargeur_donnees)

def obtenir_predictions(modele, chargeur_donnees):
    modele.eval()
    predictions = []
    with torch.no_grad():
        for caracteristiques, _ in chargeur_donnees:
            caracteristiques = caracteristiques.to(device)
            sorties = modele(caracteristiques)
            _, predit = torch.max(sorties.data, 1)
            predictions.extend(predit.cpu().numpy())
    return np.array(predictions)

# Définition du modèle
class ModeleDetectionPosition(nn.Module):
    def __init__(self, taille_entree, taille_cachee, nb_classes):
        super(ModeleDetectionPosition, self).__init__()
        self.fc1 = nn.Linear(taille_entree, taille_cachee)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(taille_cachee, nb_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def sauvegarder_metriques(chemin_fichier, metriques):
    with open(chemin_fichier, 'w') as f:
        json.dump(metriques, f)

def charger_metriques(chemin_fichier):
    if os.path.exists(chemin_fichier):
        with open(chemin_fichier, 'r') as f:
            return json.load(f)
    return None

# Modifier la fonction tracer_historique pour accepter deux séries de données
def tracer_historique(historique1, historique2, titre, ylabel, legendes, xlabel='Époques'):
    plt.plot(historique1, label=legendes[0])
    plt.plot(historique2, label=legendes[1])
    plt.title(titre)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
    
def tracer_matrice_confusion(etiquettes, predictions, classes):
    mat_conf = confusion_matrix(etiquettes, predictions)
    sns.heatmap(mat_conf, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Vérité')
    plt.xlabel('Prédiction')
    plt.show()

if __name__ == "__main__":
    
    historique_perte_train = []
    historique_precision_train = []
    historique_f1_train = []
    historique_rappel_train = []
    historique_perte_test = []
    historique_precision_test = []
    historique_f1_test = []
    historique_rappel_test = []
    
    # Ajouter des listes pour les métriques de validation
    historique_perte_val = []
    historique_precision_val = []
    historique_f1_val = []
    historique_rappel_val = []

    # Charger et prétraiter les données
    caracteristiques, etiquettes = charger_et_pretraiter_donnees(DATA)

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(caracteristiques, etiquettes, test_size=0.2, random_state=42)
    chargeur_train = creer_chargeur_donnees(X_train, y_train)
    chargeur_test = creer_chargeur_donnees(X_test, y_test)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Utilisation des ressources de ' + str(device))
    
    modele = ModeleDetectionPosition(TAILLE_ENTREE, TAILLE_CACHEE, NB_CLASSES).to(device)
        
    # Configuration de l'entraînement
    critere = nn.CrossEntropyLoss()
    optimiseur = optim.Adam(modele.parameters(), lr=TAUX_APPRENTISSAGE)
    nb_epochs = NB_EPOCHS

    # Emplacement du fichier pour stocker les métriques
    chemin_metriques = BEST_METRICS

    # Charger les métriques du meilleur modèle précédent
    meilleures_metriques = charger_metriques(chemin_metriques)
    if meilleures_metriques is None:
        meilleures_metriques = {'precision': 0, 'f1_score': 0}
    
    # Boucle d'entraînement
    for epoch in range(nb_epochs):
        modele.train()  # Mode d'entraînement
        for i, (caracteristiques, etiquettes) in enumerate(chargeur_train):
            caracteristiques = caracteristiques.to(device)
            etiquettes = etiquettes.to(device)
            
            # Passe avant
            sorties = modele(caracteristiques)
            perte = critere(sorties, etiquettes)

            # Passe arrière et optimisation
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
        
        # Évaluation après chaque époque
        perte_entrainement, precision_entrainement, f1_entrainement, rappel_entrainement = evaluer_modele(modele, chargeur_train, critere)
        print(f'Époque {epoch+1}/{nb_epochs}, Perte: {perte_entrainement:.4f}, Précision: {precision_entrainement:.4f}, F1 Score: {f1_entrainement:.4f}, Rappel: {rappel_entrainement:.4f}')
        
        historique_perte_train.append(perte_entrainement)
        historique_precision_train.append(precision_entrainement)
        historique_f1_train.append(f1_entrainement)
        historique_rappel_train.append(rappel_entrainement)
        
        perte_validation, precision_validation, f1_validation, rappel_validation = evaluer_modele(modele, chargeur_test, critere)
        historique_perte_val.append(perte_validation)
        historique_precision_val.append(precision_validation)
        historique_f1_val.append(f1_validation)
        historique_rappel_val.append(rappel_validation)
        
        if precision_entrainement > meilleures_metriques['precision'] and f1_entrainement > meilleures_metriques['f1_score']:
            meilleures_metriques = {'precision': precision_entrainement, 'f1_score': f1_entrainement}
            sauvegarder_metriques(chemin_metriques, meilleures_metriques)
            torch.save(modele.state_dict(), BEST_MODEL)
            print("Meilleur modèle sauvegardé.")

    # Sauvegarde du dernier modèle
    torch.save(modele.state_dict(), LAST_MODEL)
    print("Dernier modèle sauvegardé.")
    
    # Tracer les graphiques pour la perte et la précision
    tracer_historique(historique_perte_train, historique_perte_val, 'Perte au cours des époques', 'Perte', ['Entraînement', 'Validation'])
    tracer_historique(historique_precision_train, historique_precision_val, 'Précision au cours des époques', 'Précision', ['Entraînement', 'Validation'])

    # Évaluer le modèle sur l'ensemble de test
    perte_test, precision_test, f1_test, rappel_test = evaluer_modele(modele, chargeur_test, critere)
    print(f'Perte de test: {perte_test:.4f}, Précision de test: {precision_test:.4f}, F1 Score de test: {f1_test:.4f}, Rappel de test: {rappel_test:.4f}')

    # Obtenir les prédictions sur l'ensemble de test
    predictions_test = obtenir_predictions(modele, chargeur_test)
    tracer_matrice_confusion(y_test, predictions_test, classes=['Classe 0', 'Classe 1'])
