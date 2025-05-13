#!/bin/bash

echo "--------------------------------------------------------------------------"
echo "Script de mise à jour pour l'Interface Graphique (Listbox & Logging)"
echo "Ce script va mettre à jour vos fichiers :"
echo "  - aicompress/core.py (pour intégrer un callback de logging)"
echo "  - aicompress_gui.py (pour utiliser une Listbox et le callback)"
echo ""
echo "ATTENTION : VOS FICHIERS EXISTANTS SERONT ÉCRASÉS."
echo "‼️ Assurez-vous d'avoir fait une sauvegarde avant de continuer. ‼️"
echo ""
echo "Assurez-vous que votre environnement virtuel Python est activé si vous"
echo "comptez exécuter l'application ensuite."
echo "--------------------------------------------------------------------------"
echo ""
read -p "Appuyez sur Entrée pour continuer, ou Ctrl+C pour annuler MAINTENANT..."

# Vérifier si nous sommes (probablement) dans AICompressProject
if [ ! -d "aicompress" ] || [ ! -f "main.py" ]; then # main.py est un bon indicateur de la racine du projet
    echo "ERREUR : Ce script doit être exécuté depuis la racine de AICompressProject."
    echo "Veuillez vous déplacer dans le bon dossier et réessayer."
    exit 1
fi

# Étape 1: Mise à jour de aicompress/core.py
echo ">>> Étape 1: Mise à jour de aicompress/core.py avec le système de log_callback..."
cat << 'EOF' > aicompress/core.py
# aicompress/core.py

import os
import zipfile
import shutil
import json 

# Importez les fonctions de votre analyseur IA
try:
    from .ai_analyzer import analyze_file_content
    AI_ANALYZER_AVAILABLE = True
    # print("AI Analyzer importé avec succès depuis aicompress.core.") # Loggué par le callback maintenant
except ImportError as e:
    # Cette erreur initiale ira toujours à la console car le callback n'est pas encore configuré
    print(f"Avertissement initial (core.py): AI Analyzer (ai_analyzer.py) non trouvé ou erreur à l'import: {e}")
    AI_ANALYZER_AVAILABLE = False
    def analyze_file_content(file_path): return "ai_analyzer_unavailable"

# Gestion de rarfile
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False
    print("Avertissement initial (core.py): La bibliothèque 'rarfile' n'est pas installée.")

# ---- Définitions pour notre format ----
METADATA_FILENAME = "aicompress_metadata.json"
DEFAULT_AIC_EXTENSION = ".aic"

# Fonction de log par défaut si aucun callback n'est fourni
def _default_log(message):
    print(message)

def get_compression_settings(analysis_result, log_callback=_default_log):
    log_callback(f"[CORE] Décision de compression basée sur l'analyse IA: '{analysis_result}'")

    if analysis_result in [
        "jpeg_image", "png_image", "gif_image", "webp_image",
        "mp3_audio", "aac_audio", "ogg_vorbis_audio", "opus_audio", "flac_audio",
        "mpeg_video", "mp4_video", "webm_video", "mkv_video", "quicktime_video",
        "pdf_document",
        "gzip_archive", "bzip2_archive", "xz_archive", "lzma_archive",
    ]:
        log_callback(f"[CORE] Type '{analysis_result}' détecté. Utilisation de ZIP_STORED.")
        return zipfile.ZIP_STORED, None
    elif analysis_result in ["zip_archive", "rar_archive", "7z_archive", "tar_archive"]:
        log_callback(f"[CORE] Type '{analysis_result}' (archive conteneur) détecté. Application d'un niveau de compression DEFLATE moyen-doux (3).")
        return zipfile.ZIP_DEFLATED, 3
    elif analysis_result in ["python_script", "c_source", "java_source", "javascript_source", 
                             "html_document", "xml_document", "json_data", "css_stylesheet"]: 
        log_callback(f"[CORE] Type '{analysis_result}' (code/texte structuré) détecté. Utilisation du niveau de compression élevé (9).")
        return zipfile.ZIP_DEFLATED, 9
    elif analysis_result in ["english_text", "plain_text", "generic_text", "generic_text_heuristic"]:
        log_callback(f"[CORE] Type '{analysis_result}' (texte général) détecté. Utilisation du niveau de compression moyen-élevé (7).")
        return zipfile.ZIP_DEFLATED, 7
    elif "binary" in analysis_result or \
         analysis_result in ["file_not_found", "empty_file", "unknown", 
                              "error_during_analysis", "ai_analyzer_unavailable",
                              "error_during_fallback_analysis",
                              "unknown_binary_heuristic_null", "unknown_binary_heuristic_nonprint",
                              "unknown_binary_read_error"]:
        log_callback(f"[CORE] Type '{analysis_result}' (binaire non spécifique, inconnu ou erreur d'analyse). Utilisation d'un niveau de compression DEFLATE bas (2).")
        return zipfile.ZIP_DEFLATED, 2 
    else:
        log_callback(f"[CORE] Type '{analysis_result}' non explicitement géré. Utilisation du niveau de compression DEFLATE par défaut (6).")
        return zipfile.ZIP_DEFLATED, 6

def compress_to_aic(input_paths, output_aic_path, log_callback=_default_log):
    log_callback(f"[CORE] Début de la compression IA-modulée vers '{output_aic_path}'...")

    if not output_aic_path.endswith(DEFAULT_AIC_EXTENSION):
        log_callback(f"[CORE] Avertissement: le fichier de sortie '{output_aic_path}' n'a pas l'extension '{DEFAULT_AIC_EXTENSION}'.")

    processed_items_metadata = [] 
    try:
        with zipfile.ZipFile(output_aic_path, 'w') as zf:
            items_to_archive_details = [] 
            for item_path in input_paths:
                if not os.path.exists(item_path):
                    log_callback(f"[CORE] Attention : L'élément '{item_path}' n'existe pas et sera ignoré.")
                    processed_items_metadata.append({
                        "original_path": os.path.basename(item_path),
                        "status": "not_found", "analysis": "N/A", "compression_used": "N/A"
                    })
                    continue
                item_basename = os.path.basename(item_path)
                if os.path.isfile(item_path):
                    item_analysis_result = "ai_analyzer_unavailable"
                    if AI_ANALYZER_AVAILABLE:
                        # analyze_file_content peut avoir ses propres logs (console), ou être modifié pour un callback aussi
                        item_analysis_result = analyze_file_content(item_path) 
                    comp_type, comp_level = get_compression_settings(item_analysis_result, log_callback=log_callback)
                    items_to_archive_details.append({
                        "type": "file", "path": item_path, "arcname": item_basename,
                        "compression_type": comp_type, "compression_level": comp_level
                    })
                    processed_items_metadata.append({
                        "original_name": item_basename, "type_in_archive": "file",
                        "analysis": item_analysis_result, "size_original_bytes": os.path.getsize(item_path),
                        "compression_used": f"{'STORED' if comp_type == zipfile.ZIP_STORED else 'DEFLATED'}{f' (level {comp_level})' if comp_level is not None and comp_type == zipfile.ZIP_DEFLATED else ''}"
                    })
                elif os.path.isdir(item_path):
                    dir_metadata_entry = {
                        "original_name": item_basename, "type_in_archive": "directory",
                        "analysis": "N/A_directory", "compression_used": "N/A_directory", "files_within": []
                    }
                    for root, _, files in os.walk(item_path):
                        for file_in_dir in files:
                            full_file_path = os.path.join(root, file_in_dir)
                            archive_path_for_file = os.path.join(item_basename, os.path.relpath(full_file_path, item_path))
                            file_specific_analysis = "ai_analyzer_unavailable"
                            if AI_ANALYZER_AVAILABLE:
                                file_specific_analysis = analyze_file_content(full_file_path)
                            comp_type_f, comp_level_f = get_compression_settings(file_specific_analysis, log_callback=log_callback)
                            items_to_archive_details.append({
                                "type": "file_in_dir", "path": full_file_path, "arcname": archive_path_for_file,
                                "compression_type": comp_type_f, "compression_level": comp_level_f
                            })
                            dir_metadata_entry["files_within"].append({
                                "path_in_archive": archive_path_for_file, "analysis": file_specific_analysis,
                                "size_original_bytes": os.path.getsize(full_file_path),
                                "compression_used": f"{'STORED' if comp_type_f == zipfile.ZIP_STORED else 'DEFLATED'}{f' (level {comp_level_f})' if comp_level_f is not None and comp_type_f == zipfile.ZIP_DEFLATED else ''}"
                            })
                    processed_items_metadata.append(dir_metadata_entry)

            for item_detail in items_to_archive_details:
                if item_detail["compression_type"] == zipfile.ZIP_STORED:
                    zf.write(item_detail["path"], arcname=item_detail["arcname"], compress_type=zipfile.ZIP_STORED)
                else: 
                    zf.write(item_detail["path"], arcname=item_detail["arcname"], compress_type=zipfile.ZIP_DEFLATED, compresslevel=item_detail["compression_level"])
                log_callback(f"[CORE] Élément '{item_detail['path']}' ajouté (type: {'STORED' if item_detail['compression_type'] == zipfile.ZIP_STORED else 'DEFLATED'}, level: {item_detail['compression_level'] if item_detail['compression_type'] == zipfile.ZIP_DEFLATED else 'N/A'}) sous '{item_detail['arcname']}'.")
            
            metadata_content = {
                "aicompress_version": "0.3-alpha-ia-deep-analysis", 
                "ia_analyzer_status": "available" if AI_ANALYZER_AVAILABLE else "unavailable",
                "items_details": processed_items_metadata
            }
            metadata_str = json.dumps(metadata_content, indent=4)
            zf.writestr(METADATA_FILENAME, metadata_str)
            log_callback(f"[CORE] Fichier de métadonnées '{METADATA_FILENAME}' ajouté.")
            log_callback(f"[CORE] Compression terminée avec succès : '{output_aic_path}'")
            return True
    except FileNotFoundError:
        log_callback(f"[CORE] Erreur : Un des chemins d'entrée n'a pas été trouvé.")
        return False
    except Exception as e:
        log_callback(f"[CORE] Une erreur est survenue pendant la compression : {e}")
        import traceback
        log_callback(f"[CORE] Traceback: {traceback.format_exc()}")
        return False

def decompress_aic(aic_file_path, output_extract_path, log_callback=_default_log):
    log_callback(f"[CORE] Début de la décompression AIC de '{aic_file_path}' vers '{output_extract_path}'...")
    if not os.path.exists(aic_file_path):
        log_callback(f"[CORE] Erreur : Le fichier archive '{aic_file_path}' n'existe pas.")
        return False
    try:
        os.makedirs(output_extract_path, exist_ok=True) 
        with zipfile.ZipFile(aic_file_path, 'r') as zf:
            try:
                metadata_str = zf.read(METADATA_FILENAME)
                metadata = json.loads(metadata_str)
                log_callback("[CORE] Métadonnées de l'archive AIC :") 
                log_callback(json.dumps(metadata, indent=4))
            except KeyError:
                log_callback(f"[CORE] Avertissement : Fichier de métadonnées '{METADATA_FILENAME}' non trouvé.")
            except json.JSONDecodeError:
                log_callback(f"[CORE] Avertissement : Impossible de parser les métadonnées.")
            members_to_extract = [m for m in zf.namelist() if m != METADATA_FILENAME]
            for member in members_to_extract:
                zf.extract(member, path=output_extract_path)
                log_callback(f"[CORE] Extrait: {member}")
            log_callback(f"[CORE] Décompression AIC terminée : '{output_extract_path}'.")
            return True
    except zipfile.BadZipFile:
        log_callback(f"[CORE] Erreur : '{aic_file_path}' n'est pas un fichier ZIP valide (ou .aic corrompu).")
        return False
    except Exception as e:
        log_callback(f"[CORE] Erreur pendant la décompression AIC : {e}")
        return False

def decompress_rar(rar_file_path, output_extract_path, log_callback=_default_log):
    if not RARFILE_AVAILABLE:
        log_callback("[CORE] Erreur : Fonctionnalité RAR non disponible (rarfile ou unrar manquant).")
        return False
    log_callback(f"[CORE] Début de la décompression RAR de '{rar_file_path}' vers '{output_extract_path}'...")
    if not os.path.exists(rar_file_path):
        log_callback(f"[CORE] Erreur : Le fichier RAR '{rar_file_path}' n'existe pas.")
        return False
    try:
        os.makedirs(output_extract_path, exist_ok=True)
        with rarfile.RarFile(rar_file_path, 'r') as rf:
            rf.extractall(path=output_extract_path)
            log_callback(f"[CORE] Fichier RAR '{rar_file_path}' extrait avec succès dans '{output_extract_path}'.")
            return True
    except Exception as e: 
        log_callback(f"[CORE] Erreur pendant la décompression du RAR : {e}")
        return False

def extract_archive(archive_path, output_dir, log_callback=_default_log):
    log_callback(f"[CORE] Tentative d'extraction de '{archive_path}' vers '{output_dir}'...")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        log_callback(f"[CORE] Erreur création dossier de sortie '{output_dir}': {e}")
        return False
    if not os.path.exists(archive_path):
        log_callback(f"[CORE] Erreur : Fichier archive '{archive_path}' n'existe pas.")
        return False

    archive_type_analysis = "unknown_type"
    if AI_ANALYZER_AVAILABLE:
        archive_type_analysis = analyze_file_content(archive_path) 
        log_callback(f"[CORE] Analyse IA pour extraction de '{archive_path}': {archive_type_analysis}")

    # Prioriser l'extension pour les types connus pour l'extraction
    _, extension = os.path.splitext(archive_path)
    extension = extension.lower()

    if extension == DEFAULT_AIC_EXTENSION or archive_type_analysis == "aic_custom_format": # Supposons une future détection IA de .aic
        log_callback(f"[CORE] Format .aic détecté pour '{archive_path}'.")
        return decompress_aic(archive_path, output_dir, log_callback=log_callback)
    elif extension == ".zip" or archive_type_analysis == "zip_archive":
        log_callback(f"[CORE] Format .zip standard détecté pour '{archive_path}'.")
        try:
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(output_dir)
                log_callback(f"[CORE] Fichier .zip '{archive_path}' extrait: '{output_dir}'.")
                return True
        except Exception as e:
            log_callback(f"[CORE] Erreur décompression ZIP: {e}")
            return False
    elif extension == ".rar" or archive_type_analysis == "rar_archive":
        log_callback(f"[CORE] Format .rar détecté pour '{archive_path}'.")
        return decompress_rar(archive_path, output_dir, log_callback=log_callback)
    else:
        if zipfile.is_zipfile(archive_path): # Fallback final pour les .zip sans extension
            log_callback(f"[CORE] Type non reconnu, mais semble être un ZIP. Tentative décompression ZIP.")
            try:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(output_dir)
                    log_callback(f"[CORE] Fichier '{archive_path}' extrait (comme ZIP): '{output_dir}'.")
                    return True
            except Exception as e:
                log_callback(f"[CORE] Échec tentative décompression ZIP pour '{archive_path}': {e}")
                return False
        log_callback(f"[CORE] Format d'archive non reconnu ou non supporté pour '{archive_path}' (Analyse: {archive_type_analysis}).")
        return False
EOF
echo "aicompress/core.py mis à jour."
echo ""

# Étape 2: Mise à jour de aicompress_gui.py
echo ">>> Étape 2: Mise à jour de aicompress_gui.py avec Listbox et utilisation du log_callback..."
cat << 'EOF' > aicompress_gui.py
# aicompress_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import threading 

from aicompress.core import compress_to_aic, extract_archive, AI_ANALYZER_AVAILABLE, RARFILE_AVAILABLE

class AICompressGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("AICompress Alpha (GUI améliorée)")
        self.root.geometry("750x600") 

        self.files_to_compress = [] 

        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        compress_frame = tk.LabelFrame(main_frame, text="Compresser vers .aic", padx=10, pady=10)
        compress_frame.pack(fill=tk.X, pady=5)

        files_selection_controls_frame = tk.Frame(compress_frame)
        files_selection_controls_frame.pack(fill=tk.X, pady=(0,5))

        btn_add_files = tk.Button(files_selection_controls_frame, text="Ajouter Fichier(s)", command=self.add_files)
        btn_add_files.pack(side=tk.LEFT, padx=5)

        btn_add_folder = tk.Button(files_selection_controls_frame, text="Ajouter Dossier", command=self.add_folder)
        btn_add_folder.pack(side=tk.LEFT, padx=5)
        
        btn_clear_list = tk.Button(files_selection_controls_frame, text="Vider Liste", command=self.clear_file_list)
        btn_clear_list.pack(side=tk.LEFT, padx=5)

        # Listbox pour afficher les fichiers sélectionnés
        listbox_frame = tk.Frame(compress_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.listbox_files = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, height=6)
        self.listbox_files_scrollbar_y = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox_files.yview)
        self.listbox_files_scrollbar_x = tk.Scrollbar(listbox_frame, orient=tk.HORIZONTAL, command=self.listbox_files.xview)
        self.listbox_files.config(yscrollcommand=self.listbox_files_scrollbar_y.set, xscrollcommand=self.listbox_files_scrollbar_x.set)
        
        self.listbox_files_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_files_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.listbox_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        output_frame = tk.Frame(compress_frame) 
        output_frame.pack(fill=tk.X, pady=5) 

        lbl_output_aic = tk.Label(output_frame, text="Fichier .aic de sortie:")
        lbl_output_aic.pack(side=tk.LEFT)
        self.entry_output_aic_path = tk.Entry(output_frame, width=35)
        self.entry_output_aic_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_output_aic = tk.Button(output_frame, text="Parcourir...", command=self.browse_output_aic)
        btn_browse_output_aic.pack(side=tk.LEFT, padx=5)

        self.btn_compress_action = tk.Button(compress_frame, text="COMPRESSER", command=self.start_compression_thread, bg="lightblue", relief=tk.RAISED)
        self.btn_compress_action.pack(pady=10)

        decompress_frame = tk.LabelFrame(main_frame, text="Décompresser Archive", padx=10, pady=10)
        decompress_frame.pack(fill=tk.X, pady=10)

        source_archive_frame = tk.Frame(decompress_frame)
        source_archive_frame.pack(fill=tk.X, pady=2)
        lbl_source_archive = tk.Label(source_archive_frame, text="Fichier archive source:")
        lbl_source_archive.pack(side=tk.LEFT)
        self.entry_source_archive_path = tk.Entry(source_archive_frame, width=35)
        self.entry_source_archive_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_source_archive = tk.Button(source_archive_frame, text="Parcourir...", command=self.browse_source_archive)
        btn_browse_source_archive.pack(side=tk.LEFT, padx=5)

        dest_folder_frame = tk.Frame(decompress_frame)
        dest_folder_frame.pack(fill=tk.X, pady=2)
        lbl_dest_folder = tk.Label(dest_folder_frame, text="Dossier de destination:")
        lbl_dest_folder.pack(side=tk.LEFT)
        self.entry_dest_folder_path = tk.Entry(dest_folder_frame, width=35)
        self.entry_dest_folder_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_dest_folder = tk.Button(dest_folder_frame, text="Parcourir...", command=self.browse_dest_folder)
        btn_browse_dest_folder.pack(side=tk.LEFT, padx=5)
        
        self.btn_decompress_action = tk.Button(decompress_frame, text="DÉCOMPRESSER", command=self.start_decompression_thread, bg="lightgreen", relief=tk.RAISED)
        self.btn_decompress_action.pack(pady=10)

        log_frame = tk.LabelFrame(main_frame, text="Logs et Messages", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_message(f"AICompress GUI Initialisé. Analyseur IA: {'OK' if AI_ANALYZER_AVAILABLE else 'Non dispo.'}. Support RAR: {'OK' if RARFILE_AVAILABLE else 'Non dispo.'}")

    def log_message(self, message):
        # S'assurer que la modification de la GUI se fait dans le thread principal
        def _update_log():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END) 
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _update_log) # Planifier l'update dans la boucle Tkinter
        # print(message) # Optionnel: garder pour le debug console

    def add_files(self):
        filepaths = filedialog.askopenfilenames(title="Sélectionner fichier(s) à compresser")
        if filepaths:
            count_added = 0
            for fp in filepaths:
                if fp not in self.files_to_compress:
                    self.files_to_compress.append(fp)
                    self.listbox_files.insert(tk.END, fp) 
                    count_added += 1
            if count_added > 0:
                self.log_message(f"[GUI] {count_added} fichier(s) ajouté(s) à la liste.")
            self.listbox_files.see(tk.END)

    def add_folder(self):
        folderpath = filedialog.askdirectory(title="Sélectionner un dossier à compresser")
        if folderpath:
            if folderpath not in self.files_to_compress:
                 self.files_to_compress.append(folderpath)
                 self.listbox_files.insert(tk.END, folderpath)
                 self.log_message(f"[GUI] Dossier '{folderpath}' ajouté à la liste.")
            else:
                 self.log_message(f"[GUI] Dossier '{folderpath}' déjà dans la liste.")
            self.listbox_files.see(tk.END)
        
    def clear_file_list(self):
        self.files_to_compress = []
        self.listbox_files.delete(0, tk.END) 
        self.log_message("[GUI] Liste des fichiers à compresser vidée.")

    def browse_output_aic(self):
        filepath = filedialog.asksaveasfilename(
            title="Enregistrer le fichier AIC sous...",
            defaultextension=".aic",
            filetypes=[("AICompress Archives", "*.aic"), ("Tous les fichiers", "*.*")]
        )
        if filepath:
            self.entry_output_aic_path.delete(0, tk.END)
            self.entry_output_aic_path.insert(0, filepath)
            self.log_message(f"[GUI] Chemin de sortie AIC défini sur: {filepath}")

    def browse_source_archive(self):
        filepath = filedialog.askopenfilename(
            title="Sélectionner une archive à décompresser",
            filetypes=[("Archives Supportées", "*.aic *.zip *.rar"), ("AICompress Archives", "*.aic"), ("ZIP Archives", "*.zip"), ("RAR Archives", "*.rar"), ("Tous les fichiers", "*.*")]
        )
        if filepath:
            self.entry_source_archive_path.delete(0, tk.END)
            self.entry_source_archive_path.insert(0, filepath)
            self.log_message(f"[GUI] Archive source pour décompression: {filepath}")

    def browse_dest_folder(self):
        folderpath = filedialog.askdirectory(title="Sélectionner le dossier de destination pour l'extraction")
        if folderpath:
            self.entry_dest_folder_path.delete(0, tk.END)
            self.entry_dest_folder_path.insert(0, folderpath)
            self.log_message(f"[GUI] Dossier destination pour extraction: {folderpath}")

    def _set_buttons_state(self, state):
        if state == tk.DISABLED:
            self.btn_compress_action.config(state=tk.DISABLED, text="Compression en cours...")
            self.btn_decompress_action.config(state=tk.DISABLED, text="Décompression en cours...")
        else: # tk.NORMAL
            self.btn_compress_action.config(state=tk.NORMAL, text="COMPRESSER")
            self.btn_decompress_action.config(state=tk.NORMAL, text="DÉCOMPRESSER")


    def start_compression_thread(self):
        if not self.files_to_compress:
            messagebox.showerror("Erreur", "Aucun fichier ou dossier sélectionné pour la compression.")
            return
        output_path = self.entry_output_aic_path.get()
        if not output_path:
            messagebox.showerror("Erreur", "Veuillez spécifier un chemin pour le fichier .aic de sortie.")
            return
        
        self._set_buttons_state(tk.DISABLED)
        self.log_message(f"[GUI] Démarrage de la compression de {len(self.files_to_compress)} élément(s) vers {output_path}...")
        thread = threading.Thread(target=self.run_compression, args=(list(self.files_to_compress), output_path), daemon=True)
        thread.start()

    def run_compression(self, files_list, output_aic_path):
        try:
            success = compress_to_aic(files_list, output_aic_path, log_callback=self.log_message)
            if success:
                # Le log_callback de compress_to_aic a déjà loggué le succès
                self.root.after(0, lambda: messagebox.showinfo("Succès", f"Compression terminée !\nFichier sauvegardé sous: {output_aic_path}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Échec", "La compression a échoué. Consultez les logs."))
        except Exception as e:
            self.log_message(f"[GUI] Erreur majeure pendant la compression: {e}")
            import traceback
            self.log_message(f"[GUI] Traceback: {traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Erreur Fatale", f"Une erreur inattendue est survenue: {e}"))
        finally:
            self.root.after(0, self._set_buttons_state, tk.NORMAL)
            # self.root.after(0, self.clear_file_list) # Optionnel

    def start_decompression_thread(self):
        source_archive = self.entry_source_archive_path.get()
        dest_folder = self.entry_dest_folder_path.get()
        if not source_archive or not dest_folder:
            messagebox.showerror("Erreur", "Chemin de l'archive et dossier de destination requis.")
            return
        if not os.path.exists(source_archive):
            messagebox.showerror("Erreur", f"Fichier archive '{source_archive}' non trouvé.")
            return
            
        self._set_buttons_state(tk.DISABLED)
        self.log_message(f"[GUI] Démarrage de la décompression de {source_archive} vers {dest_folder}...")
        thread = threading.Thread(target=self.run_decompression, args=(source_archive, dest_folder), daemon=True)
        thread.start()

    def run_decompression(self, archive_path, output_dir):
        try:
            success = extract_archive(archive_path, output_dir, log_callback=self.log_message)
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Succès", f"Décompression terminée !\nFichiers extraits dans: {output_dir}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Échec", f"La décompression de '{archive_path}' a échoué."))
        except Exception as e:
            self.log_message(f"[GUI] Erreur majeure pendant la décompression: {e}")
            import traceback
            self.log_message(f"[GUI] Traceback: {traceback.format_exc()}")
            self.root.after(0, lambda: messagebox.showerror("Erreur Fatale", f"Une erreur inattendue est survenue: {e}"))
        finally:
            self.root.after(0, self._set_buttons_state, tk.NORMAL)

if __name__ == '__main__':
    app_window = tk.Tk()
    gui_app = AICompressGUI(app_window)
    app_window.mainloop()
EOF
echo "aicompress_gui.py mis à jour."
echo ""

echo "--------------------------------------------------------------------------"
echo "Mise à jour de l'interface graphique et du logging terminée."
echo "Vous pouvez maintenant exécuter l'application avec :"
echo "  python aicompress_gui.py"
echo "(Assurez-vous que votre environnement virtuel est activé et que"
echo "toutes les dépendances précédentes sont installées)."
echo "--------------------------------------------------------------------------"

exit 0