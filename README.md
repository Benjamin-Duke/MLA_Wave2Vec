# Projet MLA G14 : Wav2Vec

Juste à lancer run_finetuning.py

L'objectif du projet est de reproduire l'architecture et les expérimentations misent en place dans l'article *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*.
- Utilisation de la dataset Librispeech
- Architecture adapaté à nos contraintes
- Boucle de pré-entrainement et *fine-tuning* du modèle
- Evaluation des perfomances : WER

## Prérequis :
Les librairies nécessaires se trouve dans le fichier **requirement.txt**.  

```bash
pip install -r requirements.txt
```

## Reproduction des résultats :

### 1. Téléchargement des données
- [Corpus Librispeech](https://www.openslr.org/12) 

### 2. Pré-entraînement
- Lancer la commande suivante
```bash
python3 wave2vecTraining.py
```
### 3. Fine-Tuning
- Lancer la commande suivante
```bash
python3 fineTuningModel.py
```
### 4. Evaluation
- Lancer la commande suivante
```bash
python3 evalModel.py
```


### 5. tensorboard
- Lancer la commande suivante
```bash
tensorboard --logdir=./logs/fit --bind_all
```

## Références
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477)
- [An Illustrated Tour of Wav2vec 2.0](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html)
- [Bert : Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/pdf/1810.04805)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762)
