import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import seaborn as sns

# Chemins des fichiers et paramètres
DATA = 'preprocessed_data_merged.csv'  # Chemin vers le fichier de données prétraitées
BEST_METRICS = 'best_metrics.json'  # Chemin pour sauvegarder les meilleures métriques
BEST_MODEL = 'best_model.pt'  # Chemin pour sauvegarder le meilleur modèle
LAST_MODEL = 'last_model.pt'  # Chemin pour sauvegarder le dernier modèle
TAILLE_LOT = 16
NB_EPOCHS = 10
TAUX_APPRENTISSAGE = 2e-5 
MAX_LEN = 128

# Chargement du dataset
df = pd.read_csv(DATA)
df['label'] = df['label'].map({'Intox': 0, 'Info': 1})  # Mapping des étiquettes en numériques (0 pour Intox, 1 pour Info)

# Division du dataset en train et test
X_train, X_test, y_train, y_test = train_test_split(df['affirmation'], df['label'], test_size=0.2)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )

        return {
          'texte': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }

# Création du DataLoader
train_data = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
train_loader = DataLoader(train_data, batch_size=TAILLE_LOT, num_workers=0)

test_data = TextDataset(X_test, y_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_data, batch_size=TAILLE_LOT, num_workers=0)

# Modèle BERT pour classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.cuda()

# Optimiseur
optimizer = AdamW(model.parameters(), lr=TAUX_APPRENTISSAGE)

# Fonction d'entraînement
def train_epoch(model, data_loader, optimizer, device, n_examples):
    model = model.train()
    pertes = []
    prédictions_correctes = 0

    for d in tqdm(data_loader, total=len(data_loader)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
        )

        perte = outputs.loss
        pertes.append(perte.item())

        _, prédictions = torch.max(outputs.logits, dim=1)
        prédictions_correctes += torch.sum(prédictions == labels)

        optimizer.zero_grad()
        perte.backward()
        optimizer.step()

    return prédictions_correctes.double() / n_examples, np.mean(pertes)

# Fonction d'évaluation
def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    pertes = []
    prédictions_correctes = 0
    toutes_prédictions = []
    toutes_vraies = []

    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              labels=labels
            )

            perte = outputs.loss
            pertes.append(perte.item())

            _, prédictions = torch.max(outputs.logits, dim=1)
            prédictions_correctes += torch.sum(prédictions == labels)

            toutes_prédictions.extend(prédictions.tolist())
            toutes_vraies.extend(labels.tolist())

    f1 = f1_score(toutes_vraies, toutes_prédictions)
    recall = recall_score(toutes_vraies, toutes_prédictions)
    return prédictions_correctes.double() / n_examples, np.mean(pertes), f1, recall

def plot_confusion_matrix(conf_mat):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Étiquettes prédites')
    plt.ylabel('Étiquettes réelles')
    plt.title('Matrice de confusion')
    plt.show()

def main():

    try:
        with open(BEST_METRICS, 'r') as infile:
            meilleures_métriques = json.load(infile)
            meilleure_f1 = meilleures_métriques.get('f1', 0)
    except FileNotFoundError:
        meilleure_f1 = 0

    # Boucle d'entraînement
    historique = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': [], 'val_f1': [], 'val_recall': []}

    for époque in range(NB_EPOCHS):
        print(f'Époque {époque + 1}/{NB_EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device='cuda',
            n_examples=len(train_data)
        )

        print(f'Perte d\'entraînement {train_loss} précision {train_acc}')

        val_acc, val_loss, val_f1, val_recall = eval_model(
            model,
            test_loader,
            device='cuda',
            n_examples=len(test_data)
        )

        print(f'Perte de validation {val_loss} précision {val_acc} F1 {val_f1} Rappel {val_recall}')
        print()

        historique['train_acc'].append(train_acc.item())
        historique['train_loss'].append(train_loss.item())
        historique['val_acc'].append(val_acc.item())
        historique['val_loss'].append(val_loss.item())
        historique['val_f1'].append(val_f1.item())
        historique['val_recall'].append(val_recall.item())

        # Sauvegarde du modèle si la F1 de validation est la meilleure
        if val_f1 > meilleure_f1:
            meilleure_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL)
            meilleures_métriques = {'précision': val_acc.item(), 'f1': val_f1.item(), 'rappel': val_recall.item()}
            with open(BEST_METRICS, 'w') as fichier:
                json.dump(meilleures_métriques, fichier)
                print("Meilleur modèle trouvé.")

    # Sauvegarde du dernier modèle
    torch.save(model.state_dict(), LAST_MODEL)
    print("Dernier modèle enregistré.")

    # Affichage des courbes de perte et de précision
    plt.figure()
    plt.plot(historique['train_acc'], label='précision d\'entraînement')
    plt.plot(historique['val_acc'], label='précision de validation')
    plt.title('Précision au fil des époques')
    plt.xlabel('Époques')
    plt.ylabel('Précision')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(historique['train_loss'], label='perte d\'entraînement')
    plt.plot(historique['val_loss'], label='perte de validation')
    plt.title('Perte au fil des époques')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()

    # Calcul et affichage de la matrice de confusion
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for d in test_loader:
            input_ids = d["input_ids"].to('cuda')
            attention_mask = d["attention_mask"].to('cuda')
            labels = d["labels"].to('cuda')

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, prédictions = torch.max(outputs.logits, dim=1)
            y_pred.extend(prédictions.tolist())
            y_true.extend(labels.tolist())

    matrice_confusion = confusion_matrix(y_true, y_pred)
    print('Matrice de confusion:')
    print(matrice_confusion)

    plot_confusion_matrix(matrice_confusion)

if __name__ == '__main__':
    main()
