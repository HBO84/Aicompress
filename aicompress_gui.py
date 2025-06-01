# aicompress_gui.py
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
import os
import threading 
from datetime import datetime

# --- IMPORTS SIMPLIFIÉS depuis le package aicompress (via __init__.py) ---
try:
    from aicompress import ( # Importe directement depuis le package
        compress_to_aic,
        extract_archive,      # Devrait maintenant venir de aicompress (via aic_file_handler)
        # Flags de disponibilité
        AI_ANALYZER_AVAILABLE,
        RARFILE_AVAILABLE,
        CRYPTOGRAPHY_AVAILABLE,
        PY7ZR_SUPPORT_AVAILABLE, 
        OTA_AVAILABLE,
        # Fonctions OTA
        check_for_model_updates,
        download_and_install_model,
        # Constantes
        DEFAULT_AIC_EXTENSION
    )
    MODULES_LOADED_SUCCESSFULLY = True 
    # _gui_initial_fallback_log("INFO (GUI): Composants principaux d'AICompress importés avec succès.")
except ImportError as e_gui_pkg_import:
    MODULES_LOADED_SUCCESSFULLY = False
    print(f"ERREUR CRITIQUE (GUI): Impossible d'importer les composants principaux d'AICompress depuis __init__.py: {e_gui_pkg_import}")
    AI_ANALYZER_AVAILABLE, RARFILE_AVAILABLE, CRYPTOGRAPHY_AVAILABLE, PY7ZR_SUPPORT_AVAILABLE, OTA_AVAILABLE = False, False, False, False, False
    DEFAULT_AIC_EXTENSION = ".aic"
    def _gui_fallback_log(msg="Erreur"): print(f"[GUI_FALLBACK_LOG] {msg}")
    def compress_to_aic(i,o,p,l=_gui_fallback_log,pc=None,ce=None): l("Erreur: Moteur de compression AIC non chargé !"); return False, "AIC Components Missing"
    def extract_archive(a,o,p,l=_gui_fallback_log,pc=None,ce=None): l("Erreur: Moteur d'extraction non chargé !"); return False, "Extraction Components Missing"
    def check_for_model_updates(l=_gui_fallback_log): l("Erreur: Module OTA non chargé !"); return {}
    def download_and_install_model(n,i,l=_gui_fallback_log): l("Erreur: Module OTA non chargé !"); return False
# --- FIN IMPORTS SIMPLIFIÉS ---

# ... (le reste de votre classe AICompressGUI)

class AICompressGUI:
    # Dans aicompress_gui.py (Classe AICompressGUI)

    # Dans aicompress_gui.py (Classe AICompressGUI)

    # Dans aicompress_gui.py (à l'intérieur de la classe AICompressGUI)

    def __init__(self, root_window):
        self.root = root_window
        self.root.title(f"AICompress v1.2.3 - {DEFAULT_AIC_EXTENSION}")
        
        # Rendre la fenêtre explicitement redimensionnable (valeur par défaut, mais pour être sûr)
        self.root.resizable(True, True) 
        
        # Taille initiale (vous pouvez commenter cette ligne pour voir si Tkinter choisit une meilleure taille,
        # ou l'augmenter si des éléments sont toujours masqués)
        self.root.geometry("780x780") # Hauteur augmentée à 780 pour test

        self.files_to_compress = [] 
        self.cancel_event = threading.Event() # Événement pour l'annulation
        self.var_recursive_optimize = tk.BooleanVar() # NOUVELLE LIGNE
        # Frame principal qui s'étend pour remplir la fenêtre
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Section Compression ---
        # Ce cadre doit s'étendre pour que la Listbox à l'intérieur puisse s'étendre verticalement
        compress_frame = tk.LabelFrame(main_frame, text=f"Compresser vers {DEFAULT_AIC_EXTENSION}", padx=10, pady=10)
        compress_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5, padx=0)

        files_selection_controls_frame = tk.Frame(compress_frame)
        files_selection_controls_frame.pack(fill=tk.X, pady=(0,5)) # Ne s'étend pas verticalement
        btn_add_files = tk.Button(files_selection_controls_frame, text="Ajouter Fichier(s)", command=self.add_files)
        btn_add_files.pack(side=tk.LEFT, padx=5)
        btn_add_folder = tk.Button(files_selection_controls_frame, text="Ajouter Dossier", command=self.add_folder)
        btn_add_folder.pack(side=tk.LEFT, padx=5)
        btn_clear_list = tk.Button(files_selection_controls_frame, text="Vider Liste", command=self.clear_file_list)
        btn_clear_list.pack(side=tk.LEFT, padx=5)

        listbox_frame = tk.Frame(compress_frame) # Ce cadre contient la Listbox
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=5) # Doit s'étendre
        self.listbox_files = tk.Listbox(listbox_frame, selectmode=tk.EXTENDED, height=6) # height=6 est une taille initiale/minimale
        self.listbox_files_scrollbar_y = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.listbox_files.yview)
        self.listbox_files_scrollbar_x = tk.Scrollbar(listbox_frame, orient=tk.HORIZONTAL, command=self.listbox_files.xview)
        self.listbox_files.config(yscrollcommand=self.listbox_files_scrollbar_y.set, xscrollcommand=self.listbox_files_scrollbar_x.set)
        self.listbox_files_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox_files_scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.listbox_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) # La Listbox s'étend

        output_frame = tk.Frame(compress_frame)
        output_frame.pack(fill=tk.X, pady=5) # Ne s'étend pas verticalement
        lbl_output_aic = tk.Label(output_frame, text=f"Fichier {DEFAULT_AIC_EXTENSION}:")
        lbl_output_aic.pack(side=tk.LEFT)
        self.entry_output_aic_path = tk.Entry(output_frame, width=35)
        self.entry_output_aic_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_output_aic = tk.Button(output_frame, text="Parcourir...", command=self.browse_output_aic)
        btn_browse_output_aic.pack(side=tk.LEFT, padx=5)
        
        

        compress_options_frame = tk.Frame(compress_frame)
        
        compress_options_frame.pack(fill=tk.X, pady=(5,0)) # Ne s'étend pas verticalement
        recursive_optimize_frame = tk.Frame(compress_frame) # debut bouton recursive
        recursive_optimize_frame.pack(fill=tk.X, pady=5, anchor='w') # anchor='w' pour aligner à gauche

        self.chk_recursive_optimize = tk.Checkbutton(
            recursive_optimize_frame, 
            text="Optimiser les archives incluses (plus lent)",
            variable=self.var_recursive_optimize
        )
        self.chk_recursive_optimize.pack(side=tk.LEFT) # Fin bouton recursive
        self.var_encrypt_aic = tk.BooleanVar()
        self.chk_encrypt_aic = tk.Checkbutton(compress_options_frame, text="Protéger par mot de passe:", 
                                              variable=self.var_encrypt_aic, command=self.toggle_password_entry_compress,
                                              state=tk.NORMAL if CRYPTOGRAPHY_AVAILABLE else tk.DISABLED)
        self.chk_encrypt_aic.pack(side=tk.LEFT)
        self.entry_password_compress = tk.Entry(compress_options_frame, show="*", width=25, state=tk.DISABLED)
        self.entry_password_compress.pack(side=tk.LEFT, padx=5)
        if not CRYPTOGRAPHY_AVAILABLE: 
            tk.Label(compress_options_frame, text="(Crypto non dispo)", fg="red").pack(side=tk.LEFT)
        
        compress_action_buttons_frame = tk.Frame(compress_frame) 
        compress_action_buttons_frame.pack(pady=10) # Hauteur fixe
        self.btn_compress_action = tk.Button(compress_action_buttons_frame, text="COMPRESSER", command=self.start_compression_thread, 
                                             bg="lightblue", relief=tk.RAISED, width=15, 
                                             state=tk.NORMAL if MODULES_LOADED_SUCCESSFULLY else tk.DISABLED)
        self.btn_compress_action.pack(side=tk.LEFT, padx=10)
        self.btn_cancel_compress = tk.Button(compress_action_buttons_frame, text="Annuler", command=self.cancel_operation, 
                                             state=tk.DISABLED, width=10)
        self.btn_cancel_compress.pack(side=tk.LEFT, padx=10)
        
        # --- Section Décompression ---
        # Ce cadre ne doit pas s'étendre verticalement, juste horizontalement
        decompress_frame = tk.LabelFrame(main_frame, text="Décompresser Archive", padx=10, pady=10)
        decompress_frame.pack(side=tk.TOP, fill=tk.X, pady=10) 
        
        source_archive_frame = tk.Frame(decompress_frame)
        source_archive_frame.pack(fill=tk.X, pady=2)
        lbl_source_archive = tk.Label(source_archive_frame, text="Archive source:")
        lbl_source_archive.pack(side=tk.LEFT)
        self.entry_source_archive_path = tk.Entry(source_archive_frame, width=35)
        self.entry_source_archive_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_source_archive = tk.Button(source_archive_frame, text="Parcourir...", command=self.browse_source_archive)
        btn_browse_source_archive.pack(side=tk.LEFT, padx=5)
        
        dest_folder_frame = tk.Frame(decompress_frame)
        dest_folder_frame.pack(fill=tk.X, pady=2)
        lbl_dest_folder = tk.Label(dest_folder_frame, text="Dossier destination:")
        lbl_dest_folder.pack(side=tk.LEFT)
        self.entry_dest_folder_path = tk.Entry(dest_folder_frame, width=35)
        self.entry_dest_folder_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_dest_folder = tk.Button(dest_folder_frame, text="Parcourir...", command=self.browse_dest_folder)
        btn_browse_dest_folder.pack(side=tk.LEFT, padx=5)
        
        password_frame_decompress = tk.Frame(decompress_frame)
        password_frame_decompress.pack(fill=tk.X, pady=2)
        lbl_password_decompress = tk.Label(password_frame_decompress, text="Mot de passe (si requis):")
        lbl_password_decompress.pack(side=tk.LEFT)
        self.entry_password_decompress = tk.Entry(password_frame_decompress, show="*", width=30)
        self.entry_password_decompress.pack(side=tk.LEFT, padx=5)

        decompress_action_buttons_frame = tk.Frame(decompress_frame) 
        decompress_action_buttons_frame.pack(pady=10) # Hauteur fixe
        self.btn_decompress_action = tk.Button(decompress_action_buttons_frame, text="DÉCOMPRESSER", command=self.start_decompression_thread, 
                                               bg="lightgreen", relief=tk.RAISED, width=15, 
                                               state=tk.NORMAL if MODULES_LOADED_SUCCESSFULLY else tk.DISABLED)
        self.btn_decompress_action.pack(side=tk.LEFT, padx=10)
        self.btn_cancel_decompress = tk.Button(decompress_action_buttons_frame, text="Annuler", command=self.cancel_operation, 
                                               state=tk.DISABLED, width=10)
        self.btn_cancel_decompress.pack(side=tk.LEFT, padx=10)
        
        # --- Barre de Progression ---
        # Ce cadre ne doit pas s'étendre verticalement
        progress_section_frame = tk.Frame(main_frame)
        progress_section_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 5)) 
        self.lbl_progress = tk.Label(progress_section_frame, text="Progression:")
        self.lbl_progress.pack(side=tk.LEFT, padx=(0,5))
        self.progress_bar = ttk.Progressbar(progress_section_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True) # Le progressbar s'étend horizontalement
        self.lbl_progress_percentage = tk.Label(progress_section_frame, text="0%", width=4)
        self.lbl_progress_percentage.pack(side=tk.LEFT, padx=(5,0))
            
        # --- Utilitaires (OTA) ---
        # Ce cadre ne doit pas s'étendre verticalement
        extra_actions_frame = tk.LabelFrame(main_frame, text="Utilitaires", padx=10, pady=10)
        extra_actions_frame.pack(side=tk.TOP, fill=tk.X, pady=5) 
        self.btn_check_ota_updates = tk.Button(extra_actions_frame, text="Vérifier MàJ Modèles IA", 
                                               command=self.run_check_ota_updates, 
                                               state=tk.NORMAL if OTA_AVAILABLE and MODULES_LOADED_SUCCESSFULLY else tk.DISABLED)
        self.btn_check_ota_updates.pack(side=tk.LEFT, padx=5)
        if not OTA_AVAILABLE: 
            tk.Label(extra_actions_frame, text="(Module OTA non dispo)", fg="red").pack(side=tk.LEFT)

        # --- Barre de Statut (en bas) ---
        status_bar_frame = tk.Frame(main_frame, bd=1, relief=tk.SUNKEN)
        status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X) # Fixé en bas, s'étend horizontalement

        # --- Zone de Log (doit prendre l'espace vertical restant) ---
        # Packé en dernier parmi les éléments TOP, pour qu'il prenne l'espace restant
        log_frame = tk.LabelFrame(main_frame, text="Logs et Messages", padx=10, pady=10)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5) 
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED) # height=10 est une taille minimale
        self.log_text.pack(fill=tk.BOTH, expand=True) # Le ScrolledText s'étend
        
        # Message d'initialisation dans la barre de statut
        status_text_init = f"Crypto:{'OK' if CRYPTOGRAPHY_AVAILABLE else 'X'} | IA An.:{'OK' if AI_ANALYZER_AVAILABLE else 'X'} | " \
                           f"OTA:{'OK' if OTA_AVAILABLE else 'X'} | RAR:{'OK' if RARFILE_AVAILABLE else 'X'} | " \
                           f"7z:{'OK' if PY7ZR_SUPPORT_AVAILABLE else 'X'}"
        self.lbl_status_bar = tk.Label(status_bar_frame, text=status_text_init, anchor=tk.W)
        self.lbl_status_bar.pack(side=tk.LEFT)
        
        self.log_message(f"AICompress GUI (v1.2.3 Resizable Attempt) Initialisé. {status_text_init}")
        if not MODULES_LOADED_SUCCESSFULLY: 
            messagebox.showerror("Erreur Critique", "Modules principaux non chargés. Vérifiez la console.")

        # --- Configuration pour le redimensionnement de la fenêtre principale et diagnostic ---
        # Forcer Tkinter à calculer les tailles requises par tous les widgets
        self.root.update_idletasks() 
        
        # Obtenir la taille minimale requise
        min_w = self.root.winfo_reqwidth()
        min_h = self.root.winfo_reqheight()
        
        # Définir cette taille comme la taille minimale de la fenêtre
        self.root.minsize(min_w, min_h) 
        self.log_message(f"[GUI_DEBUG] Taille minimale requise (reqwidth, reqheight) définie à : {min_w}x{min_h}")
        
        # Vérifier et logger l'état actuel de la propriété resizable
        resizable_status = self.root.resizable()
        self.log_message(f"[GUI_DEBUG] État resizable final de la fenêtre (width, height): {resizable_status}")
        if not (resizable_status[0] and resizable_status[1]):
            self.log_message("[GUI_DEBUG] AVERTISSEMENT: La fenêtre n'est PAS marquée comme redimensionnable par Tkinter après initialisation !")

    def toggle_password_entry_compress(self): # Identique
        if self.var_encrypt_aic.get() and CRYPTOGRAPHY_AVAILABLE: self.entry_password_compress.config(state=tk.NORMAL)
        else: self.entry_password_compress.config(state=tk.DISABLED); self.entry_password_compress.delete(0, tk.END)

    def log_message(self, message): # Identique
        def _update_log():
            if hasattr(self, 'log_text') and self.log_text.winfo_exists(): self.log_text.config(state=tk.NORMAL); self.log_text.insert(tk.END, str(message) + "\n"); self.log_text.see(tk.END); self.log_text.config(state=tk.DISABLED)
        if hasattr(self, 'root') and self.root.winfo_exists(): self.root.after(0, _update_log)
        print(message) 
    
    def add_files(self):
        filepaths = filedialog.askopenfilenames(title="Sélectionner fichier(s) à compresser")
        if filepaths:
            count_added = 0
            for fp_gui in filepaths: 
                normalized_fp = os.path.normpath(fp_gui)
                if normalized_fp not in self.files_to_compress:
                    # --- Instructions sur des lignes séparées et indentées ---
                    self.files_to_compress.append(normalized_fp)
                    self.listbox_files.insert(tk.END, normalized_fp)
                    count_added +=1
            if count_added > 0:
                self.log_message(f"[GUI] {count_added} fichier(s) ajouté(s).")
            if self.listbox_files.winfo_exists():
                self.listbox_files.see(tk.END)

    def add_folder(self):
        folderpath = filedialog.askdirectory(title="Sélectionner un dossier à compresser")
        if folderpath:
            normalized_folderpath = os.path.normpath(folderpath)
            if normalized_folderpath not in self.files_to_compress:
                # --- Instructions sur des lignes séparées et indentées ---
                self.files_to_compress.append(normalized_folderpath)
                self.listbox_files.insert(tk.END, normalized_folderpath)
                self.log_message(f"[GUI] Dossier '{normalized_folderpath}' ajouté.")
            else:
                self.log_message(f"[GUI] Dossier '{normalized_folderpath}' déjà listé.")
            if self.listbox_files.winfo_exists():
                self.listbox_files.see(tk.END)
        
    def clear_file_list(self): # Identique
        self.files_to_compress = []; 
        if self.listbox_files.winfo_exists(): self.listbox_files.delete(0, tk.END); 
        self.log_message("[GUI] Liste vidée.")

    def browse_output_aic(self): # Identique
        fp = filedialog.asksaveasfilename(title="Enregistrer sous...",defaultextension=DEFAULT_AIC_EXTENSION,filetypes=[(f"AICompress (*{DEFAULT_AIC_EXTENSION})",f"*{DEFAULT_AIC_EXTENSION}"),("Tous","*.*")])
        if fp:
            b,e=os.path.splitext(fp);
            if e.lower()!=DEFAULT_AIC_EXTENSION.lower():fp=b+DEFAULT_AIC_EXTENSION
            if self.entry_output_aic_path.winfo_exists(): self.entry_output_aic_path.delete(0,tk.END); self.entry_output_aic_path.insert(0,fp); 
            self.log_message(f"[GUI] Sortie AIC: {fp}")

    def browse_source_archive(self): # Identique
        ft=[("Archives Supportées",f"*{DEFAULT_AIC_EXTENSION} *.zip *.rar"+(" *.7z" if PY7ZR_SUPPORT_AVAILABLE else "")), (f"AICompress (*{DEFAULT_AIC_EXTENSION})",f"*{DEFAULT_AIC_EXTENSION}"),("ZIP","*.zip"),("RAR","*.rar")]
        if PY7ZR_SUPPORT_AVAILABLE: ft.append(("7-Zip","*.7z"))
        ft.append(("Tous","*.*")); fp=filedialog.askopenfilename(title="Sélectionner archive",filetypes=ft)
        if fp: 
            if self.entry_source_archive_path.winfo_exists(): self.entry_source_archive_path.delete(0,tk.END); self.entry_source_archive_path.insert(0,fp); 
            self.log_message(f"[GUI] Archive source: {fp}")

    def browse_dest_folder(self): # Identique
        fp=filedialog.askdirectory(title="Sélectionner dossier destination")
        if fp: 
            if self.entry_dest_folder_path.winfo_exists(): self.entry_dest_folder_path.delete(0,tk.END); self.entry_dest_folder_path.insert(0,fp); 
            self.log_message(f"[GUI] Dossier destination: {fp}")

    def _set_buttons_state(self, operation_in_progress_type=None): # 'compress', 'decompress', 'ota', ou None pour tout réactiver
        # Désactiver tous les boutons d'action principaux
        if self.btn_compress_action.winfo_exists(): self.btn_compress_action.config(state=tk.DISABLED)
        if self.btn_decompress_action.winfo_exists(): self.btn_decompress_action.config(state=tk.DISABLED)
        if self.btn_check_ota_updates.winfo_exists(): self.btn_check_ota_updates.config(state=tk.DISABLED)
        # Désactiver tous les boutons d'annulation
        if self.btn_cancel_compress.winfo_exists(): self.btn_cancel_compress.config(state=tk.DISABLED)
        if self.btn_cancel_decompress.winfo_exists(): self.btn_cancel_decompress.config(state=tk.DISABLED)

        if not hasattr(self, "_default_button_texts"): self._default_button_texts = {}
        if self.btn_compress_action not in self._default_button_texts: self._default_button_texts[self.btn_compress_action] = "COMPRESSER"
        if self.btn_decompress_action not in self._default_button_texts: self._default_button_texts[self.btn_decompress_action] = "DÉCOMPRESSER"
        if self.btn_check_ota_updates not in self._default_button_texts: self._default_button_texts[self.btn_check_ota_updates] = "Vérifier MàJ Modèles IA"


        if operation_in_progress_type == 'compress':
            if self.btn_compress_action.winfo_exists(): self.btn_compress_action.config(text="Compression...")
            if self.btn_cancel_compress.winfo_exists(): self.btn_cancel_compress.config(state=tk.NORMAL)
        elif operation_in_progress_type == 'decompress':
            if self.btn_decompress_action.winfo_exists(): self.btn_decompress_action.config(text="Décompression...")
            if self.btn_cancel_decompress.winfo_exists(): self.btn_cancel_decompress.config(state=tk.NORMAL)
        elif operation_in_progress_type == 'ota':
            if self.btn_check_ota_updates.winfo_exists(): self.btn_check_ota_updates.config(text="Vérification...")
            # Pas de bouton d'annulation pour OTA pour l'instant
        else: # Réactiver tous les boutons d'action, s'assurer que les boutons d'annulation sont désactivés
            if self.btn_compress_action.winfo_exists(): self.btn_compress_action.config(state=tk.NORMAL if MODULES_LOADED_SUCCESSFULLY else tk.DISABLED, text=self._default_button_texts[self.btn_compress_action])
            if self.btn_decompress_action.winfo_exists(): self.btn_decompress_action.config(state=tk.NORMAL if MODULES_LOADED_SUCCESSFULLY else tk.DISABLED, text=self._default_button_texts[self.btn_decompress_action])
            if self.btn_check_ota_updates.winfo_exists(): self.btn_check_ota_updates.config(state=tk.NORMAL if OTA_AVAILABLE and MODULES_LOADED_SUCCESSFULLY else tk.DISABLED, text=self._default_button_texts[self.btn_check_ota_updates])
            # Les boutons Annuler sont déjà désactivés par défaut ou par cancel_operation()

    def cancel_operation(self):
        if self.cancel_event:
            self.log_message("[GUI] Demande d'annulation de l'opération en cours...")
            self.cancel_event.set()
            # Désactiver les boutons d'annulation pour éviter les clics multiples
            if hasattr(self, 'btn_cancel_compress') and self.btn_cancel_compress.winfo_exists():
                self.btn_cancel_compress.config(state=tk.DISABLED)
            if hasattr(self, 'btn_cancel_decompress') and self.btn_cancel_decompress.winfo_exists():
                self.btn_cancel_decompress.config(state=tk.DISABLED)

    def reset_progress_bar(self): # Identique
        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
            if self.progress_bar['mode'] == 'indeterminate': self.progress_bar.stop()
            self.progress_bar.config(mode='determinate'); self.progress_bar["value"] = 0; self.progress_bar["maximum"] = 100 
        if hasattr(self, 'lbl_progress_percentage') and self.lbl_progress_percentage.winfo_exists(): self.lbl_progress_percentage.config(text="0%")

    def update_progress_bar(self, current_value, max_value): # Identique
        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
            if self.progress_bar['mode'] == 'indeterminate': self.progress_bar.stop(); self.progress_bar.config(mode='determinate')
            if max_value > 0:
                self.progress_bar["maximum"] = max_value; self.progress_bar["value"] = current_value
                percentage = int((current_value / max_value) * 100)
                if hasattr(self, 'lbl_progress_percentage') and self.lbl_progress_percentage.winfo_exists(): self.lbl_progress_percentage.config(text=f"{percentage}%")
            elif max_value == 0 and current_value == 0:
                 self.progress_bar["value"] = 0
                 if hasattr(self, 'lbl_progress_percentage') and self.lbl_progress_percentage.winfo_exists(): self.lbl_progress_percentage.config(text="0%")

    def handle_progress_update(self, current_item_count, total_items): # Identique
        if self.root.winfo_exists(): self.root.after(0, self.update_progress_bar, current_item_count, total_items)

    def start_compression_thread(self):
        if not MODULES_LOADED_SUCCESSFULLY: 
            messagebox.showerror("Erreur", "Modules AICompress non chargés !")
            return
        if not self.files_to_compress: 
            messagebox.showerror("Erreur", "Aucun fichier/dossier à compresser.")
            return
        output_path = self.entry_output_aic_path.get()
        if not output_path: 
            messagebox.showerror("Erreur", "Chemin de sortie requis.")
            return
        
        base, ext = os.path.splitext(output_path)
        if ext.lower() != DEFAULT_AIC_EXTENSION.lower():
            output_path = base + DEFAULT_AIC_EXTENSION
            self.log_message(f"[GUI] Extension {DEFAULT_AIC_EXTENSION} appliquée. Sortie: {output_path}")
            if self.entry_output_aic_path.winfo_exists():
                self.entry_output_aic_path.delete(0,tk.END)
                self.entry_output_aic_path.insert(0,output_path)
        
        password_for_compression = None
        if self.var_encrypt_aic.get():
            if not CRYPTOGRAPHY_AVAILABLE: messagebox.showerror("Erreur", "Crypto non dispo."); return
            password_for_compression = self.entry_password_compress.get()
            if not password_for_compression: messagebox.showerror("Erreur", "Mdp requis si chiffré."); return
            if len(password_for_compression) < 4: messagebox.showwarning("Attention", "Mdp court.")
        
        # --- NOUVEAU : Lire la valeur de la case à cocher ---
        optimize_recursively = self.var_recursive_optimize.get() # Renvoie True si cochée, False sinon
        
        self.cancel_event.clear()
        self._set_buttons_state(operation_in_progress_type='compress')
        self.reset_progress_bar() 
        
        log_msg_recursive = " (Mode Optimisation d'Archives: Activé)" if optimize_recursively else ""
        self.log_message(f"[GUI] Compression vers {output_path} (Chiffré: {'Oui' if password_for_compression else 'Non'}){log_msg_recursive}...")
        
        thread = threading.Thread(target=self.run_compression, 
                                  args=(list(self.files_to_compress), 
                                        output_path, 
                                        password_for_compression, 
                                        self.handle_progress_update,
                                        optimize_recursively), # AJOUT DU NOUVEL ARGUMENT
                                  daemon=True)
        thread.start()

    def run_compression(self, files_list, output_aic_path, password_compress, 
                        progress_callback_gui, optimize_recursively): # AJOUT DU NOUVEL ARGUMENT
        status_msg_for_popup = "Erreur inconnue compression."
        success_flag = False 
        try:
            success_flag, status_msg_from_core = compress_to_aic(
                files_list, 
                output_aic_path, 
                password_compress=password_compress, 
                log_callback=self.log_message, 
                progress_callback=progress_callback_gui,
                cancel_event=self.cancel_event,
                recursively_optimize=optimize_recursively # AJOUT : Passer l'option au backend
            )
            
            # ... (reste de la fonction run_compression comme avant) ...
            status_msg_for_popup = str(status_msg_from_core)
            if self.cancel_event.is_set():
                self.log_message("[GUI] Compression annulée par l'utilisateur (détecté dans run_compression).")
                status_msg_for_popup = "Compression annulée."
                success_flag = False
            
            def _show_msg_comp():
                if not self.root.winfo_exists(): return
                if success_flag: messagebox.showinfo("Succès", f"Compression terminée !\nSauvegardé: {output_aic_path}")
                elif status_msg_for_popup == "Compression annulée.": messagebox.showinfo("Annulé", "La compression a été annulée.")
                else: messagebox.showerror("Échec", f"Compression échouée.\nMotif: {status_msg_for_popup}")
            if self.root.winfo_exists(): self.root.after(0, _show_msg_comp)

        except Exception as e: 
            self.log_message(f"[GUI] Erreur majeure compression: {e}"); import traceback; self.log_message(f"[GUI] Traceback: {traceback.format_exc()}"); 
            status_msg_for_popup = str(e)
            def _show_fatal_comp(msg=status_msg_for_popup):
                if self.root.winfo_exists(): messagebox.showerror("Erreur Fatale", f"Erreur:\n{msg}")
            if self.root.winfo_exists(): self.root.after(0, _show_fatal_comp)
        finally:
            if self.root.winfo_exists(): 
                self.root.after(0, self._set_buttons_state, None) # Réactiver tous les boutons

    def start_decompression_thread(self):
        if not MODULES_LOADED_SUCCESSFULLY: 
            messagebox.showerror("Erreur", "Modules AICompress non chargés !")
            return

        source_archive = self.entry_source_archive_path.get()
        dest_folder = self.entry_dest_folder_path.get()
        password = self.entry_password_decompress.get()

        if not source_archive or not dest_folder:
            messagebox.showerror("Erreur", "Archive et destination requises.")
            return
        if not os.path.exists(source_archive):
            messagebox.showerror("Erreur", f"Archive '{source_archive}' non trouvée.")
            return
        
        self.cancel_event.clear() 
        self._set_buttons_state(operation_in_progress_type='decompress') 
        self.reset_progress_bar() # Remet en mode 'determinate' par défaut

        source_archive_lower = source_archive.lower()
        is_aic_file = source_archive_lower.endswith(DEFAULT_AIC_EXTENSION.lower())
        is_zip_file = source_archive_lower.endswith(".zip")
        is_rar_file = source_archive_lower.endswith(".rar")
        
        callback_for_backend = None
        # Le mode passé au thread run_decompression pour qu'il sache comment nettoyer
        progress_mode_for_thread_run = "indeterminate" 

        if is_aic_file or is_zip_file or is_rar_file: 
            callback_for_backend = self.handle_progress_update
            progress_mode_for_thread_run = "determinate"
            self.log_message(f"[GUI_DEBUG] start_decomp: Mode progression pour {os.path.basename(source_archive)}: determinate")
            if hasattr(self,'progress_bar') and self.progress_bar.winfo_exists():
                 self.progress_bar.config(mode='determinate')
                 # La barre sera mise à jour par le callback avec le bon max
        else: # Pour .7z et autres formats inconnus, on utilise le mode indéterminé
            progress_mode_for_thread_run = "indeterminate" 
            self.log_message(f"[GUI_DEBUG] start_decomp: Mode progression pour {os.path.basename(source_archive)}: indeterminate. Démarrage animation.")
            if hasattr(self,'progress_bar') and self.progress_bar.winfo_exists():
                self.progress_bar.config(mode='indeterminate')
                self.progress_bar.start(30) 
            if hasattr(self,'lbl_progress_percentage') and self.lbl_progress_percentage.winfo_exists():
                self.lbl_progress_percentage.config(text="En cours...")
            self.root.update_idletasks() 
        
        self.log_message(f"[GUI] Décompression de {source_archive} (Mdp: {'Oui' if password else 'Non'})...")
        
        thread = threading.Thread(target=self.run_decompression, 
                                  args=(source_archive, dest_folder, 
                                        password if password else None, 
                                        callback_for_backend, 
                                        progress_mode_for_thread_run), # Passer le mode correct
                                  daemon=True)
        thread.start()

    def run_decompression(self, archive_path, output_dir, password, 
                          progress_callback_to_backend, actual_progress_mode_from_start): # Nom de paramètre clarifié
        self.log_message(f"[GUI_DEBUG] run_decompression: actual_progress_mode='{actual_progress_mode_from_start}', callback_is_None={progress_callback_to_backend is None}")
        status_code_for_popup = "Erreur inconnue décompression."
        is_pwd_error_flag_list = [False] 
        success_flag = False
        try:
            success_flag, status_code_from_core = extract_archive(
                archive_path, output_dir, password=password, 
                log_callback=self.log_message,
                progress_callback=progress_callback_to_backend,
                cancel_event=self.cancel_event 
            )
            status_code_for_popup = str(status_code_from_core)

            if self.cancel_event.is_set(): 
                self.log_message(f"[GUI] Décompression pour '{os.path.basename(archive_path)}' annulée.")
                status_code_for_popup = "Décompression annulée."
                success_flag = False 
            
            if not (not success_flag and any(err_code_part in status_code_for_popup for err_code_part in ["PasswordError", "PasswordNeeded", "BadRarFileOrPassword", "PasswordError7z", "Bad7zFileOrPassword"])):
                def _show_decomp_result_message_final(): # Renommé pour unicité
                    if not self.root.winfo_exists(): return
                    if success_flag: messagebox.showinfo("Succès", f"Décompression terminée: {output_dir}")
                    elif status_code_for_popup == "Décompression annulée.": messagebox.showinfo("Annulé", "La décompression a été annulée.")
                    else: messagebox.showerror("Échec", f"Décompression de '{os.path.basename(archive_path)}' échouée.\nMotif: {status_code_for_popup}")
                if self.root.winfo_exists(): self.root.after(0, _show_decomp_result_message_final)
            
            if not success_flag and any(err_code_part in status_code_for_popup for err_code_part in ["PasswordError", "PasswordNeeded", "BadRarFileOrPassword", "PasswordError7z", "Bad7zFileOrPassword"]):
                is_pwd_error_flag_list[0] = True
                self.log_message(f"[GUI] Échec (mot de passe?): {status_code_for_popup} pour {archive_path}.")
                if self.root.winfo_exists(): 
                    self.root.after(0, self.prompt_for_password_and_retry_decompression, archive_path, output_dir, actual_progress_mode_from_start)
        
        except Exception as e: 
            self.log_message(f"[GUI] Erreur majeure décompression: {e}"); import traceback; self.log_message(f"[GUI] Traceback: {traceback.format_exc()}"); 
            status_code_for_popup = str(e); success_flag = False
            def _show_fe_d_run(msg=status_code_for_popup):
                if self.root.winfo_exists(): messagebox.showerror("Erreur Fatale", f"Erreur:\n{msg}")
            if self.root.winfo_exists(): self.root.after(0, _show_fe_d_run)
        
        finally:
            if self.root.winfo_exists():
                self.root.after(0, self._finalize_gui_after_decompression, 
                                actual_progress_mode_from_start, 
                                success_flag, 
                                is_pwd_error_flag_list[0])

    def _finalize_gui_after_decompression(self, progress_mode_that_ran, operation_succeeded, is_retrying_password):
        """Nettoie l'état de la GUI après une tentative de décompression."""
        if not self.root.winfo_exists(): return
        self.log_message(f"[GUI_DEBUG] Finalize: mode_ran='{progress_mode_that_ran}', succeeded={operation_succeeded}, retrying_pwd={is_retrying_password}")

        if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
            current_bar_mode = self.progress_bar['mode']
            self.log_message(f"[GUI_DEBUG] Finalize: Current bar mode before stop/config: '{current_bar_mode}'")
            if current_bar_mode == 'indeterminate' or progress_mode_that_ran == 'indeterminate': # Être plus explicite
                self.log_message(f"[GUI_DEBUG] Finalize: Stopping indeterminate bar (mode_that_ran='{progress_mode_that_ran}').")
                self.progress_bar.stop()
            self.progress_bar.config(mode='determinate') # Toujours remettre en déterminé à la fin
            self.log_message(f"[GUI_DEBUG] Finalize: Bar mode after config: '{self.progress_bar['mode']}'")
        
        if not is_retrying_password: 
            if operation_succeeded:
                if hasattr(self, 'progress_bar') and self.progress_bar.winfo_exists():
                    current_max = self.progress_bar.cget('maximum')
                    if current_max == 0 or current_max == 1: 
                        self.update_progress_bar(1, 1) 
                    elif current_max > 0 : 
                        self.update_progress_bar(current_max, current_max)
                    else: self.reset_progress_bar()
            else: # Échec ou Annulation (et pas un retry de mdp)
                self.log_message("[GUI_DEBUG] Finalize: Operation failed or cancelled, resetting progress bar.")
                self.reset_progress_bar() 
            
            self.log_message("[GUI_DEBUG] Finalize: Setting buttons to NORMAL state.")
            self._set_buttons_state(None) 
        else:
            self.log_message("[GUI_DEBUG] Finalize: Is retrying password, buttons/bar state managed by prompt_for_password.")

    def run_check_ota_updates(self): # ... (Identique)
        if not OTA_AVAILABLE: self.log_message("[GUI] Module OTA non dispo."); messagebox.showwarning("OTA Indisponible", "Module OTA non chargé."); return
        self.log_message("[GUI] Vérification MàJ modèles IA..."); self.cancel_event.clear(); self._set_buttons_state(operation_in_progress_type='ota') 
        thread = threading.Thread(target=self._perform_ota_check_and_prompt, daemon=True); thread.start()

    def _perform_ota_check_and_prompt(self): # ... (Identique)
        updates_info_for_gui=[]; models_to_update_actions=[]; action_taken_flag_ota = [False] 
        try:
            updates_available = check_for_model_updates(log_callback=self.log_message)
            if self.cancel_event.is_set(): self.log_message("[GUI] Vérification OTA annulée."); return
            if not updates_available: updates_info_for_gui.append("Aucune nouvelle MàJ de modèle IA disponible.")
            else:
                updates_info_for_gui.append("MàJ de modèles IA disponibles:"); 
                for mn,si in updates_available.items(): updates_info_for_gui.append(f"  - {mn}: v{si.get('latest_version')} ({si.get('description','N/A')})"); models_to_update_actions.append((mn,si))
                if models_to_update_actions: updates_info_for_gui.append("\nInstaller ces MàJ ?")
            if self.root.winfo_exists(): self.root.after(0, self._show_ota_results_and_prompt_install, updates_info_for_gui, models_to_update_actions, action_taken_flag_ota)
        except Exception as e: 
            self.log_message(f"[GUI] Erreur vérification OTA: {e}"); import traceback; self.log_message(f"[GUI] Traceback OTA: {traceback.format_exc()}"); 
            if self.root.winfo_exists(): self.root.after(0, lambda err=e: messagebox.showerror("Erreur OTA", f"Erreur MàJ: {err}"))
        finally: 
            if not action_taken_flag_ota[0] and self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, None)

    def _show_ota_results_and_prompt_install(self, info_messages, models_to_install, action_taken_flag): # ... (Identique)
        if not self.root.winfo_exists(): self._set_buttons_state(None); return
        full_message = "\n".join(info_messages); should_ask = bool(models_to_install); user_response = False
        if should_ask:
            user_response = messagebox.askyesno("Mises à Jour Modèles IA", full_message)
            if user_response:
                action_taken_flag[0] = True; self.log_message("[GUI] Installation MàJ acceptée."); 
                if self.btn_check_ota_updates.winfo_exists(): self.btn_check_ota_updates.config(text="Installation MàJ...") # Le _set_buttons_state gérera ça aussi
                thread = threading.Thread(target=self._perform_model_installations, args=(models_to_install,), daemon=True); thread.start()
            else: self.log_message("[GUI] Installation MàJ refusée."); self._set_buttons_state(None)
        else: messagebox.showinfo("Mises à Jour Modèles IA", full_message); self._set_buttons_state(None)

    def _perform_model_installations(self, models_to_install): # ... (Identique)
        all_s = True; any_a = bool(models_to_install)
        for mn, si in models_to_install:
            if self.cancel_event.is_set(): self.log_message(f"[GUI] Installation MàJ {mn} annulée."); all_s = False; break
            self.log_message(f"[GUI] Installation MàJ pour {mn}..."); success = download_and_install_model(mn,si,log_callback=self.log_message) # Supposer que download_and_install_model gère cancel_event
            if success: self.log_message(f"[GUI] MàJ {mn} réussie !")
            else: self.log_message(f"[GUI] ÉCHEC MàJ {mn}."); all_s = False
        if self.root.winfo_exists():
            if any_a and not self.cancel_event.is_set(): # N'afficher que si non annulé
                if all_s: self.root.after(0, lambda: messagebox.showinfo("Mises à Jour IA", "Modèles mis à jour !"))
                else: self.root.after(0, lambda: messagebox.showwarning("Mises à Jour IA", "Certaines MàJ ont échoué ou ont été annulées."))
        if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, None)

if __name__ == '__main__':
    app_window = tk.Tk()
    gui_app = AICompressGUI(app_window)
    app_window.mainloop()