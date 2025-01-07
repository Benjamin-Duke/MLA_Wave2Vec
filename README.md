# Projet MLA G14 : Wav2Vec

Il faut télécharger les données pour entraîner le LM Transformer : 
https://www.openslr.org/11/
-> norm et vocab
Créer un folder LM_data et mettre ces deux fichiers dedans.
D'abord :
python train_lm.py \
    --data_dir LM_data \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --max_length 512 \
    --log_dir lm_runs \
    --checkpoint_dir lm_checkpoints

Pour entrainer le transformer, ensuite :
python evaluate.py \
    --model_path /path/to/finetuned/wav2vec/model \
    --lm_path lm_checkpoints/best_model.pt \
    --vocab_path LM_data/librispeech-vocab.txt \
    --beam_size 100 \
    --lm_weight 0.3 \
    --word_score -1.0 \
    --output_file evaluation_results.txt


Pour évaluer



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
