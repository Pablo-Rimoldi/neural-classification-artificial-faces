# Risultati esperimenti DL

## 1. `ERP_DL_Complete.ipynb`

### Modelli testati
- **EEGNet**: architettura compatta specifica per low-data EEG, con blocco temporale, convoluzione spaziale depthwise e convoluzione separabile.
- **GNN** (simplificato): matrice di adiacenza apprendibile, convoluzione temporale e pooling adattivo.
- **CNN 1D** (semplificata): due blocchi convoluzionali 1D con filtri ridotti e kernel calibrati per ERP.
- **LSTM** (semplificato): rete ricorrente con hidden ridotto, progettata per catturare componenti tardive del segnale.
- **EEGNet-SSL**: encoder pre-addestrato su autoencoder senza label, seguito da una testa classificatrice minimale.

### Strategia di validazione e regolarizzazione
- Nested CV con `GroupKFold(5)` esterno e `GroupShuffleSplit(3)` interno.
- Ottimizzazione iperparametri con Optuna (10 trial per fold).
- Metrica interna: loss di validazione.
- Training con augmentation EEG-specifica:
  - jitter temporale,
  - channel dropout,
  - gaussian noise,
  - amplitude scaling,
  - mixup intra-batch.
- Early stopping leggero basato sulla validazione interna.

### Risultati numerici osservati
- EEGNet: **58.8% ± 11.6% accuracy**, **AUC 0.620**
- GNN: **61.0% ± 5.4% accuracy**, **AUC 0.649**
- CNN 1D: **57.0% ± 3.7% accuracy**, **AUC 0.616**
- LSTM: **53.9% ± 7.3% accuracy**, **AUC 0.501**
- EEGNet-SSL: **54.1% ± 5.4% accuracy**, **AUC 0.597**

### Contesto del confronto
- La migliore performance media emersa nel notebook è stata quella del modello **GNN** con **61.0% accuracy** e **AUC 0.649**.
- Tutti i modelli sono valutati con la stessa pipeline di Nested CV e con la stessa augmentation EEG-specifica.
- Il dataset è piccolo e rumoroso, quindi la deviazione standard delle accuracy è un indicatore importante della stabilità.

## 2. `GNN_optimization.ipynb`

### Modello e abilitazioni
- **GNN Advanced**: architettura basata su matrice di adiacenza apprendibile con opzioni per:
  - inizializzazione `adj_init` (`uniform`, `identity`, `random`),
  - normalizzazione `adj_norm` (`none`, `softmax`, `sigmoid`),
  - 1 o 2 layer convoluzionali temporali con residual connection,
  - dropout e pooling adattivo.

### Strategia di ricerca iperparametri
- Nested CV con `GroupKFold(5)` esterno e `GroupShuffleSplit(3)` interno.
- Optuna con **CMA-ES** su 50 trial per fold.
- **MedianPruner** per interrompere i trial peggiori e risparmiare tempo.
- Spazio iperparametri esteso a 10 variabili:
  - architettura: `temp_filters`, `kernel_size`, `n_layers`, `adj_init`, `adj_norm`
  - regolarizzazione: `dropout`
  - ottimizzazione: `lr`, `wd`, `scheduler`
  - training: `batch_size`, `epochs`, `mixup_alpha`

### Risultati numerici osservati
- Baseline GNN: **61.0% ± 5.4% accuracy**, **AUC 0.649**.
- GNN Advanced (best): **64.1% ± 5.7% accuracy**, **AUC 0.678**.
  - Fold details: **75%**, **60%**, **59%**, **63%**, **63%**.
- GNN Ensemble Top-3: **55.1% ± 5.2% accuracy**, **AUC 0.665**.
  - Fold-level ensemble accuracy: **65.0%**, **55.0%**, **50.0%**, **52.6%**, **52.6%**.

### Confronto e osservazioni
- Il tuning avanzato con Optuna/CMA-ES ha migliorato la accuracy media rispetto alla baseline di circa **3.1 punti percentuali**.
- L’ensemble Top-3 ha mostrato una stabilità leggermente inferiore e una media più bassa rispetto al modello best, ma è utile per analizzare la robustezza.
- Il notebook contiene anche:
  - un confronto finale tra baseline, GNN Advanced e ensemble,
  - report di classificazione e confusion matrix,
  - bias di genere maschi/femmine per le predizioni dei modelli,
  - analisi Optuna delle scelte iperparametriche più efficaci.

### Note aggiuntive
- Il notebook fornisce anche un’analisi delle correlazioni tra iperparametri e val_loss, oltre a visualizzare la distribuzione dei migliori trial.
- La pipeline è progettata per migliorare la stabilità e ridurre la varianza su un dataset EEG molto piccolo.
