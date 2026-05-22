import json

def main():
    notebook_path = 'notebooks/dl_test_byclass.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']

    final_conclusions_md = """---
# Conclusioni Finali: XAI e Interpretazione Neurofisiologica

Alla luce dell'addestramento e dei risultati corretti della **Fase 2 (Explainable AI - XAI)** sulla **TemporalGraphNet (GNN) ottimizzata**, possiamo trarre conclusioni profonde su come il modello (e per estensione il cervello) distingua un volto umano da uno generato dall'IA.

### 1. Dinamica Temporale: Dalla Percezione al Giudizio (Permutation & Saliency)
L'analisi temporale tramite **Saliency Maps** e **Permutation Importance** offre un quadro chiaro sullo sviluppo temporale della classificazione:
- **~200-300 ms (P200/N250)**: Le Saliency Maps mostrano una prima banda di forte attivazione. Coincide con la decodifica strutturale profonda del viso.
- **~500-600 ms**: Ulteriore forte attivazione visibile nelle Saliency Maps, legato a processi cognitivi prolungati di valutazione.
- **>700 ms (LPP - Late Positive Potential)**: Le mappe di salienza esplodono nell'ultima fase. La **Permutation Importance** per finestre temporali corrobora totalmente questo dato biologico: la finestra **W5 (164-205 frame, ~640-800ms)** è emersa come *l'unica* macro-finestra essenziale per non far crollare drasticamente l'accuracy del classificatore. Privato di tale finestra, il modello fallisce. Le fasi precoci (W1-W4) mostrano drop negativi quando corrotte, ad indicare che il modello considera l'elaborazione sensoriale di base (luce, contrasto, forme primarie in W1-W2) come fattori confundenti ai fini della distinzione tra reale e generato. Il riconoscimento del falso avviene nella fase di valutazione cognitiva tardiva e cosciente (LPP).

### 2. Le Aree Cerebrali Critiche e il Ruolo del Genere (Channel Importance)
Attraverso la **Channel Importance**, individuiamo da dove proviene il segnale decisivo e quali metadati supportano il network:
- **I Lobi Parietali e Temporo-Parietali**: Le feature spaziali più critiche in assoluto (che massimizzano la caduta delle performance se omesse) sono i canali **P3, TP8 e P4**. Questa è una solida conferma: le aree parietali e temporali destre/sinistre sono hub cruciali per il riconoscimento facciale avanzato e la valutazione spaziale-strutturale.
- **Canali Frontali e Confusori**: Sorprendentemente, se i canali AFF4h, AFF1h, AFF2h portano un lieve vantaggio, canali frontali o occipitali visivi primari come AFF3h, O1 e O2, se oscurati, *migliorano* le performance (drop negativo). La rete ritiene che i segnali in queste aree (es. processamento primario occipitale o task cognitivi extra frontali) aggiungano rumore.
- **L'importanza dei Metadati (SubjectSEX e PCA)**: Un risultato fondamentale emerge dal fatto che la feature **SubjectSEX** ha un'importanza positiva. Questo indica che la rete utilizza l'informazione sul sesso biologico del soggetto EEG per "calibrare" la propria interpretazione dei segnali, suggerendo che esistano pattern o baseline ERP diverse per genere nel processare volti IA in questo dataset. Le PC estratte hanno importanza marginale (PCA_Occipital in lieve positivo), ad eccezione delle componenti PCA_Frontal e PCA_Parietal che generano confusione.

### 3. Spazio-Tempo: Isole di Importanza (Heatmap)
La **Mappa Spaziotemporale (Canale × Tempo)** concilia le due dimensioni e localizza "isole" spazio-temporali estremamente limitate in cui il segnale reale si separa dal falso:
- **P4 tra 234-312 ms**: Il picco massimo positivo di drop si trova sul canale parietale destro (P4) nella finestra N250/P300. In questo istante critico, le differenze strutturali vengono analizzate. 
- **L'utilità di calibrazione precoce del SubjectSEX**: Il sesso del soggetto si rivela utile in una finestra precoce e intermedia (**78-234 ms**), fungendo probabilmente da "prior" per il modello prima che l'onda P300 e la componente tardiva entrino in gioco per chiudere la classificazione.
- **Rumore tardivo e disattivazione frontale**: Spicca un picco fortemente negativo di "disturbo" fornito dal canale AF3 nella parte tardiva dell'epoca (546-624 ms) ed in PO9 a 234-312 ms. L'inibizione o il filtraggio di questi "falsi allarmi" cognitivi è essenziale per la GNN.

### Sintesi e Impatto Scientifico
Il successo empirico della TemporalGraphNet ottimizzata trova conferma neurobiologica negli strumenti XAI moderni: 
Identificare la differenza tra intelligenza artificiale e realtà non è un processo di "primo sguardo" (le classiche onde precoci confondono e distraggono l'agente classificatore), bensì dipende da:
1. Un pivot strutturale a cavallo dei 250-300 ms nelle aree parietali (**P3, P4, TP8**).
2. Un supporto di calibrazione pre-calcolato sulle differenze intrinseche di genere del percettore (importante utilizzo del **SubjectSEX**).
3. Una complessa valutazione cognitiva a lungo termine: la fase che "blinda" storicamente il processo decisionale risiede nei potenziali tardivi post-600ms (LPP)."""

    # Replace Cell 36
    cells[36]['source'] = [line + '\n' for line in final_conclusions_md.split('\n')]
    if cells[36]['source']:
        cells[36]['source'][-1] = cells[36]['source'][-1].rstrip('\n')

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Conclusions correctly updated in {notebook_path} based on new channels and XAI maps.")

if __name__ == '__main__':
    main()
