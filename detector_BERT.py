import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Paramètres du modèle
MODELE_BERT_CHEMIN = 'last_model.pt'  # Mettez ici le chemin de votre fichier .pt

# Vérifier si CUDA est disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Créer une instance du modèle BERT
# Si vous avez sauvegardé la configuration du modèle, chargez-la ici. Sinon, utilisez un modèle prédéfini comme 'bert-base-uncased'
modele_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Charger l'état du modèle
etat_modele = torch.load(MODELE_BERT_CHEMIN, map_location=device)
modele_bert.load_state_dict(etat_modele)

# Envoyer le modèle au dispositif et le mettre en mode évaluation
modele_bert.to(device)
modele_bert.eval()

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predire_bert(texte):
    # Tokenisation et préparation des entrées
    inputs = tokenizer(texte, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

    # Prédiction
    with torch.no_grad():
        sorties = modele_bert(**inputs)
    
    # Calcul des probabilités
    probas = torch.nn.functional.softmax(sorties.logits, dim=1)
    classe = probas.argmax(dim=1).item()
    proba = probas[0, classe].item()
    return classe, proba

# Interface utilisateur
if __name__ == "__main__":
    texte_utilisateur = input("Entrez une phrase pour la détection : ")
    classe, proba = predire_bert(texte_utilisateur)
    resultat = "Info" if classe == 0 else "Intox"  # Adaptez selon le codage de vos classes
    print(f"Résultat : {resultat}, Probabilité : {proba:.4f}")
