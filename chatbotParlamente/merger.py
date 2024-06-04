import os
from pypdf import PdfMerger


#Questa funzione merge_pdfs prende come input il percorso della cartella che contiene i PDF (folder_path) 
#e il percorso dove salvare il PDF unito (output_path). La funzione utilizza os.walk per attraversare 
#tutti i file nella cartella e unisce solo quelli che terminano con .pdf (ignorando maiuscole/minuscole).
def merge_pdfs(folder_path, output_path):
    merger = PdfMerger()

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                merger.append(file_path)
    
    merger.write(output_path)
    merger.close()

# Esempio di utilizzo:
folder_path = 'RegistroCameraDeputatiPorgetto'  # Cartella con i PDF all'interno del progetto
output_path = 'merged_RegistroCmeraDeputati.pdf'  # Nome del file unito all'interno del progetto
merge_pdfs(folder_path, output_path)


