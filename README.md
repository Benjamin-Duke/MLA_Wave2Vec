# Projet MLA G14 : Wav2Vec

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
- Lancer les commandes suivantes :
```bash
cd wave2vec_pretrain
python train_wav2vec.py
```
### 3. Fine-Tuning
- Lancer les commandes suivantes :
```bash
cd wave2vec_fine_tuning
python run_finetuning.py --pretrained_path /path/to/bestmodellast.pt
```
- Si vous souhaitez changer les paramètres du fine tuning (les valeurs ci-dessous sont celles par défaut):
```bash
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_steps 50000 \
    --classifier_steps 10000 \
    --log_dir finetuning_runs \
    --checkpoint_dir finetuning_checkpoints
```
### 4. Evaluation
- Pour lancer l'entrainement du décodeur
```bash
cd wave2vec_eval
python script_name.py --text_file path_to_your_text_file --output_path path_to_save_model --batch_size 64 --num_epochs 10 --learning_rate 0.0003
```
- Pour lancer l'évaluation du modèle
```bash
cd wave2vec_eval
python script_name.py --model_path path_to_your_model --lm_path path_to_your_language_model --vocab_path path_to_vocabulary_file --beam_size 100 --lm_weight 0.3 --word_score -1.0 --output_file evaluation_results.txt --data_dir /path/to/librispeech/data --cache_dir /path/to/cache
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
