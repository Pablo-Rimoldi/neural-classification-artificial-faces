import json
import sys

def main():
    notebook_path = 'notebooks/dl_test_byclass.ipynb'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']

    # New Final Conclusions Markdown
    final_conclusions_md = """---
# Conclusioni Finali: XAI e Interpretazione Neurofisiologica

Alla luce dell'addestramento e dei risultati della **Fase 2 (Explainable AI - XAI)** sulla **TemporalGraphNet (GNN) ottimizzata**, possiamo trarre conclusioni profonde su come il cervello distingua un volto umano da uno generato dall'IA.

### 1. Dinamica Temporale: Dalla Percezione al Giudizio (IG)
L'analisi avanzata con **Integrated Gradients (IG)** e **Permutation Importance** offre una fotografia ad altissima risoluzione di ciò che il modello (e quindi il cervello) "osserva". Il profilo temporale mostra chiari picchi di attenzione:
- **~230 ms (P200)**: Prima decodifica strutturale profonda e analisi delle distanze facciali.
- **~380-400 ms (N400)**: Massimo picco di attivazione. Questo coincide tipicamente con processi legati all'**incongruenza semantica**: il cervello nota qualcosa di "insolito", subliminale o artificiale nei tratti del volto emergente, tipico del fenomeno dell'*Uncanny Valley*.
- **~480-530 ms e protrazione >640 ms (LPP - Late Positive Potential)**: Fasi tardive associate ad una valutazione cognitiva superiore e consapevole.
La **Permutation Importance per finestre temporali** supporta radicalmente questo ultimo dato: **la finestra W5 (640-800ms)** è emersa come l'unica macro-finestra essenziale per non far crollare l'accuracy del classificatore. Ciò indica che la decisione finale AI vs Umano si cristallizza nelle fasi di valutazione tardiva, non al primo sguardo. Le prime finestre temporali presentano valori negativi di "drop", il che significa che l'elaborazione visiva base (es. luce, contrasto) confonde il modello rispetto al reale task discriminativo.

### 2. Le Aree Cerebrali Critiche (Organizzazione Spaziale)
Attraverso la valutazione della **Channel Importance**, individuiamo da dove proviene il segnale decisivo. I canali il cui oscuramento peggiora criticamente il classificatore si trovano nelle aree specializzate per la visione ad alto livello e il riconoscimento facciale:
- I canali più discriminatori in assoluto risultano **P10, PO7, P3, e TP7** (Aree Parieto-Occipitali e Temporo-Parietali).
- Al contrario, la corruzione di svariate zone frontali ed extra-visive (es. AFF3h, AF4) migliora (drop in deviazione negativo) le performance della rete. Questo implica un'iper-focalizzazione sulle componenti visive posteriori da parte della GNN: i lobi frontali ospitano probabilmente solo "rumore" o attivazione cognitiva generica/indipendente in questo specifico compito dicotomico.

### 3. Spazio-Tempo: Finestre Specifiche (Heatmap)
La **Mappa Spaziotemporale (Canale × Tempo)** concilia queste estrazioni in "isole informative":
- C'è un picco vitale altamente localizzato nell'area destra **PO8 tra 150-234 ms**, corrispondente al complesso decodificatore d'elezione per i volti **(N170 / P200)**. In questo istante rapidissimo, per la prima volta i segnali veri vs. falsi si ramificano in modo che il modello trova utile estrarre feature locali.

### Sintesi e Impatto Scientifico
Il successo empirico della TemporalGraphNet ottimizzata (Accuracy **74.0%**, bias di genere trascurabile Δ3.0%) si sposa con una validazione neurobiologica fortissima offerta dagli strumenti XAI. 
Il riconoscimento di un "falso generativo" cerebrale non è appannaggio delle prime cortecce visive, bensì scaturisce da una sequenza complessa:
1. Una rilevazione di pattern morfologici microscopici (PO8 in N170/P200).
2. Un brusco senso di incongruenza semantica visiva (picco N400).
3. Una complessa architettura di valutazione ed attribuzione cognitiva di lungo termine (LPP), corroborata nelle areee parietali.

Questi risultati non solo dimostrano la validità tecnica delle Graph Neural Networks nel processare dati EEG-ERP, ma forniscono conferme sul modo in cui la corteccia umana adatta meccanismi noti (incongruenza, percezione visiva superiore) ad un fenomeno sociologico e tecnologico completamente nuovo come le Intelligenze Artificiali Generative."""

    # Replace Cell 36 (which is currently a placeholder for XAI Summary)
    cells[36]['source'] = final_conclusions_md.split('\n')
    # add \n to each line except the last
    for i in range(len(cells[36]['source'])-1):
        cells[36]['source'][i] += '\n'
        
    # Also optionally let's fix Cell 24 heading from "Conclusioni" to "Conclusioni Intermedie (Risultati Classificazione)"
    for i, line in enumerate(cells[24]['source']):
        if line.startswith("# Conclusioni"):
            cells[24]['source'][i] = line.replace("# Conclusioni", "# Conclusioni Intermedie (Risultati Classificazione)")

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"Conclusions written to {notebook_path} successfully!")

if __name__ == '__main__':
    main()
