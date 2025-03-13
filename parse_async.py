import os
import glob
import time
import concurrent.futures
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

# Funzione per elaborare un singolo file PDF
# Deve essere definita a livello di modulo per funzionare con multiprocessing
def process_pdf(pdf_file, output_folder):
    """Elabora un singolo file PDF in un processo separato"""
    # Importa le dipendenze all'interno della funzione per evitare problemi con il pickle
    from llama_cloud_services import LlamaParse
    from llama_index.core import SimpleDirectoryReader
    
    start_time = time.time()
    print(f"Elaborazione di: {pdf_file}")
    
    # Ottieni il nome del file senza l'estensione
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    md_file_path = os.path.join(output_folder, base_name + '.md')
    
    try:
        # Configura il parser per questo processo
        # Il parser è già impostato per generare markdown
        parser = LlamaParse(result_type="markdown")
        file_extractor = {".pdf": parser}
        
        # Elabora il singolo PDF
        documents = SimpleDirectoryReader(
            input_files=[pdf_file], 
            file_extractor=file_extractor
        ).load_data()
        
        # Combina tutti i chunk in un unico testo
        # Il contenuto è già in formato markdown
        combined_text = ""
        for doc in documents:
            combined_text += doc.text + "\n\n"
        
        # Salva il testo combinato nel file di output markdown
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        elapsed = time.time() - start_time
        print(f"Salvato: {md_file_path} (tempo: {elapsed:.2f} secondi)")
        return True
        
    except Exception as e:
        print(f"Errore durante l'elaborazione di {pdf_file}: {str(e)}")
        return False

def main():
    # Imposta l'inizio del timer per l'intero processo
    total_start_time = time.time()
    
    # Cerca tutti i file PDF nella cartella "data"
    pdf_files = glob.glob('data/*.pdf')
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("Nessun file PDF trovato nella cartella 'data'.")
        return

    # Crea la cartella "output" se non esiste
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Trovati {total_files} file PDF da elaborare.")
    
    # Determina il numero ottimale di worker
    # Di solito, il numero di CPU disponibili è un buon punto di partenza
    max_workers = min(os.cpu_count(), total_files)
    print(f"Elaborazione in parallelo con {max_workers} processi.")
    
    # Usa ProcessPoolExecutor per elaborare i file in parallelo
    # Ogni file viene elaborato in un processo separato
    successful = 0
    failed = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Crea un dizionario di future per tenere traccia dei processi
        future_to_file = {
            executor.submit(process_pdf, pdf_file, output_folder): pdf_file
            for pdf_file in pdf_files
        }
        
        # Elabora i risultati man mano che diventano disponibili
        for future in concurrent.futures.as_completed(future_to_file):
            pdf_file = future_to_file[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Eccezione durante l'elaborazione di {pdf_file}: {str(e)}")
                failed += 1
    
    # Calcola il tempo totale di elaborazione
    total_elapsed = time.time() - total_start_time
    
    # Mostra il riepilogo finale
    print(f"\nElaborazione completata in {total_elapsed:.2f} secondi.")
    print(f"File elaborati con successo: {successful}/{total_files}")
    
    if failed > 0:
        print(f"File non elaborati: {failed}")
        
    if successful > 0:
        print(f"Tempo medio per file: {total_elapsed / successful:.2f} secondi")

if __name__ == "__main__":
    main()