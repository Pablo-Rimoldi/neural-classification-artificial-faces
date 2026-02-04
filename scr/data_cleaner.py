import pandas as pd
import os

def clean_ml_files(source_dir='data\Files for ML', target_dir='data/file_cleaned'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Cartella creata: {target_dir}")

    if not os.path.exists(source_dir):
        print(f"Errore: La cartella sorgente '{source_dir}' non esiste.")
        return

    for filename in os.listdir(source_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(source_dir, filename)
            try:
                df = pd.read_csv(file_path, sep='\s+', engine='python')
                new_filename = filename.replace('.txt', '.csv')
                save_path = os.path.join(target_dir, new_filename)
                df.to_csv(save_path, index=False)            
                print(f"Processato: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Errore nel processare {filename}: {e}")
    print("\nOperazione completata!")


if __name__ == "__main__":
    clean_ml_files()