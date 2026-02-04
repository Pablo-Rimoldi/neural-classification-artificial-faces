# Seeing the Uncanny: Neural Classification of Artificial Faces
## Project Overview
This project investigates the neural representation of AI-generated hyper-realistic faces compared to real human faces. Despite behavioral difficulty in distinguishing them, our goal is to use Machine Learning and Deep Learning to classify brain responses (ERP) recorded via EEG.

### Scientific Context
- **P.I.:** Prof. Alice Mado Proverbio (Univ. Milano-Bicocca)
- **Supervision:** Prof. Claudia Casellato (Univ. Pavia)
- **Team:** Pablo Rimoldi, Tommaso Godino, Andrea De Paola, Giacomo Colombo

## Technical Specifications
- **Data Source:** 128-channel EEG (10/5% system), 512 Hz sampling rate.
- **Stimuli:** 440 male/female faces (Real vs. GAN-generated).
- **Signal Processing:** 
  - Bandpass filter: 0.01-70 Hz.
  - Notch filter: 50 Hz.
  - Epochs: -100 ms to 800 ms.
  - Reference: Common Average Reference (CAR).

## Classification Task
We aim to classify EEG trials into 4 categories:
- **Code 50:** AI Male
- **Code 60:** AI Female
- **Code 70:** Real Male
- **Code 80:** Real Female

### Key Features
Focus on the **200-600 ms** time window and specific electrodes:
`O1, O2, PO9, PO10, TP7, TP8, P3, P4, AF3, AF4, AFF1h, AFF2h, AFF3h, AFF4h`.


## Quick Start
1. **clone  repository**
   \`\`\`bash
   git clone (https://github.com/Pablo-Rimoldi/neural-classification-artificial-faces)
   \`\`\`


2. **Install dependencies**:
   '''bash
   pip install pandas
   '''

2. **Run cleaning**:
   \`\`\`bash
   python src/data_cleaner.py
   \`\`\`