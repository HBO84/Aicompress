# aicompress_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
import os
import threading 

from aicompress.core import (compress_to_aic, extract_archive, 
                             AI_ANALYZER_AVAILABLE_FLAG_FROM_MODULE as AI_ANALYZER_AVAILABLE,
                             RARFILE_AVAILABLE, DEFAULT_AIC_EXTENSION,
                             CRYPTOGRAPHY_AVAILABLE)
try:
    from aicompress.ota_updater import check_for_model_updates, download_and_install_model
    OTA_AVAILABLE = True
except ImportError as e_ota:
    print(f"AVERTISSEMENT (GUI): ota_updater.py non trouvé ou erreur import: {e_ota}")
    OTA_AVAILABLE = False
    def check_for_model_updates(log_callback): 
        if callable(log_callback): log_callback("[GUI] Module OTA non disponible.")
        return {}
    def download_and_install_model(name, info, log_callback): 
        if callable(log_callback): log_callback("[GUI] Module OTA non disponible.")
        return False

class AICompressGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title(f"AICompress Alpha (Orchestrator Ready) - {DEFAULT_AIC_EXTENSION}")
        self.root.geometry("780x680") 

        self.files_to_compress = [] 

        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Section Compression ---
        compress_frame = tk.LabelFrame(main_frame, text=f"Compresser vers {DEFAULT_AIC_EXTENSION}", padx=10, pady=10)
        compress_frame.pack(fill=tk.X, pady=5)
        
        files_selection_controls_frame = tk.Frame(compress_frame)
        files_selection_controls_frame.pack(fill=tk.X, pady=(0,5))
        btn_add_files = tk.Button(files_selection_controls_frame, text="Ajouter Fichier(s)", command=self.add_files)
        btn_add_files.pack(side=tk.LEFT, padx=5)
        btn_add_folder = tk.Button(files_selection_controls_frame, text="Ajouter Dossier", command=self.add_folder)
        btn_add_folder.pack(side=tk.LEFT, padx=5)
        btn_clear_list = tk.Button(files_selection_controls_frame, text="Vider Liste", command=self.clear_file_list)
        btn_clear_list.pack(side=tk.LEFT, padx=5)

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
        lbl_output_aic = tk.Label(output_frame, text=f"Fichier {DEFAULT_AIC_EXTENSION}:")
        lbl_output_aic.pack(side=tk.LEFT)
        self.entry_output_aic_path = tk.Entry(output_frame, width=35)
        self.entry_output_aic_path.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn_browse_output_aic = tk.Button(output_frame, text="Parcourir...", command=self.browse_output_aic)
        btn_browse_output_aic.pack(side=tk.LEFT, padx=5)

        compress_options_frame = tk.Frame(compress_frame)
        compress_options_frame.pack(fill=tk.X, pady=(5,0))
        self.var_encrypt_aic = tk.BooleanVar()
        self.chk_encrypt_aic = tk.Checkbutton(compress_options_frame, text="Protéger par mot de passe:", 
                                              variable=self.var_encrypt_aic, command=self.toggle_password_entry_compress,
                                              state=tk.NORMAL if CRYPTOGRAPHY_AVAILABLE else tk.DISABLED)
        self.chk_encrypt_aic.pack(side=tk.LEFT)
        self.entry_password_compress = tk.Entry(compress_options_frame, show="*", width=25, state=tk.DISABLED)
        self.entry_password_compress.pack(side=tk.LEFT, padx=5)
        if not CRYPTOGRAPHY_AVAILABLE: 
            tk.Label(compress_options_frame, text="(Crypto non dispo)", fg="red").pack(side=tk.LEFT)
        
        self.btn_compress_action = tk.Button(compress_frame, text="COMPRESSER", command=self.start_compression_thread, bg="lightblue", relief=tk.RAISED, width=15)
        self.btn_compress_action.pack(pady=10)

        # --- Section Décompression ---
        decompress_frame = tk.LabelFrame(main_frame, text="Décompresser Archive", padx=10, pady=10)
        decompress_frame.pack(fill=tk.X, pady=10)
        
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
        
        self.btn_decompress_action = tk.Button(decompress_frame, text="DÉCOMPRESSER", command=self.start_decompression_thread, bg="lightgreen", relief=tk.RAISED, width=15)
        self.btn_decompress_action.pack(pady=10)
        
        # --- Cadre pour les actions supplémentaires (OTA) ---
        extra_actions_frame = tk.LabelFrame(main_frame, text="Utilitaires", padx=10, pady=10)
        extra_actions_frame.pack(fill=tk.X, pady=5)

        self.btn_check_ota_updates = tk.Button(extra_actions_frame, text="Vérifier MàJ Modèles IA", 
                                               command=self.run_check_ota_updates,
                                               state=tk.NORMAL if OTA_AVAILABLE else tk.DISABLED)
        self.btn_check_ota_updates.pack(side=tk.LEFT, padx=5)
        if not OTA_AVAILABLE:
            tk.Label(extra_actions_frame, text="(Module OTA non dispo)", fg="red").pack(side=tk.LEFT)

        # --- Zone de Log ---
        log_frame = tk.LabelFrame(main_frame, text="Logs et Messages", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_message(f"AICompress GUI (v0.9 Orch) Initialisé. Crypto:{'OK' if CRYPTOGRAPHY_AVAILABLE else 'X'}. IA An.:{'OK' if AI_ANALYZER_AVAILABLE else 'X'}. OTA:{'OK' if OTA_AVAILABLE else 'X'}.")

    def toggle_password_entry_compress(self):
        if self.var_encrypt_aic.get() and CRYPTOGRAPHY_AVAILABLE:
            self.entry_password_compress.config(state=tk.NORMAL)
        else:
            self.entry_password_compress.config(state=tk.DISABLED)
            self.entry_password_compress.delete(0, tk.END)

    def log_message(self, message):
        def _update_log():
            if self.log_text.winfo_exists(): 
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, str(message) + "\n")
                self.log_text.see(tk.END) 
                self.log_text.config(state=tk.DISABLED)
        if self.root.winfo_exists():
            self.root.after(0, _update_log)
        print(message) 
    
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
            if self.listbox_files.winfo_exists():
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
            if self.listbox_files.winfo_exists():
                self.listbox_files.see(tk.END)
        
    def clear_file_list(self):
        self.files_to_compress = []
        if self.listbox_files.winfo_exists():
            self.listbox_files.delete(0, tk.END) 
        self.log_message("[GUI] Liste des fichiers à compresser vidée.")

    def browse_output_aic(self):
        filepath = filedialog.asksaveasfilename(
            title="Enregistrer le fichier AIC sous...",
            defaultextension=DEFAULT_AIC_EXTENSION,
            filetypes=[(f"AICompress Archives (*{DEFAULT_AIC_EXTENSION})", f"*{DEFAULT_AIC_EXTENSION}"), 
                       ("Tous les fichiers", "*.*")]
        )
        if filepath:
            base, ext = os.path.splitext(filepath)
            if ext.lower() != DEFAULT_AIC_EXTENSION.lower():
                filepath = base + DEFAULT_AIC_EXTENSION
            
            self.entry_output_aic_path.delete(0, tk.END)
            self.entry_output_aic_path.insert(0, filepath)
            self.log_message(f"[GUI] Chemin de sortie AIC défini sur: {filepath}")

    def browse_source_archive(self):
        filepath = filedialog.askopenfilename(
            title="Sélectionner une archive à décompresser",
            filetypes=[("Archives Supportées", f"*{DEFAULT_AIC_EXTENSION} *.zip *.rar *.7z"), 
                       (f"AICompress Archives (*{DEFAULT_AIC_EXTENSION})", f"*{DEFAULT_AIC_EXTENSION}"), 
                       ("ZIP Archives", "*.zip"), 
                       ("RAR Archives", "*.rar"), 
                       ("Tous les fichiers", "*.*")]
        )
        if filepath:
            self.entry_source_archive_path.delete(0,tk.END)
            self.entry_source_archive_path.insert(0,filepath)
            self.log_message(f"[GUI] Archive source pour décompression: {filepath}")

    def browse_dest_folder(self):
        folderpath = filedialog.askdirectory(title="Sélectionner le dossier de destination pour l'extraction")
        if folderpath:
            self.entry_dest_folder_path.delete(0,tk.END)
            self.entry_dest_folder_path.insert(0,folderpath)
            self.log_message(f"[GUI] Dossier destination pour extraction: {folderpath}")

    def _set_buttons_state(self, state):
        target_buttons = [self.btn_compress_action, self.btn_decompress_action, self.btn_check_ota_updates]
        # Garder une référence aux textes originaux pour chaque bouton
        if not hasattr(self, "_default_button_texts"):
            self._default_button_texts = {btn: btn.cget("text") for btn in target_buttons if btn}
        
        busy_texts_map = {
            self.btn_compress_action: "Compression...",
            self.btn_decompress_action: "Décompression...",
            self.btn_check_ota_updates: "Vérification..."
        }

        for btn in target_buttons:
            if btn and btn.winfo_exists():
                default_text = self._default_button_texts.get(btn, "Action")
                busy_text = busy_texts_map.get(btn, "En cours...")
                
                new_text = busy_text if state == tk.DISABLED else default_text
                btn.config(state=state, text=new_text)

    def start_compression_thread(self):
        if not self.files_to_compress: messagebox.showerror("Erreur", "Aucun fichier/dossier."); return
        output_path = self.entry_output_aic_path.get()
        if not output_path: messagebox.showerror("Erreur", "Chemin de sortie requis."); return

        base, ext = os.path.splitext(output_path)
        if ext.lower() != DEFAULT_AIC_EXTENSION.lower():
            output_path = base + DEFAULT_AIC_EXTENSION
            self.log_message(f"[GUI] Extension {DEFAULT_AIC_EXTENSION} appliquée. Sortie: {output_path}")
            self.entry_output_aic_path.delete(0, tk.END); self.entry_output_aic_path.insert(0, output_path)
        
        password_for_compression = None
        if self.var_encrypt_aic.get():
            if not CRYPTOGRAPHY_AVAILABLE: messagebox.showerror("Erreur", "Crypto non dispo."); return
            password_for_compression = self.entry_password_compress.get()
            if not password_for_compression: messagebox.showerror("Erreur", "Mdp requis si chiffré."); return
            if len(password_for_compression) < 4: messagebox.showwarning("Attention", "Mdp court.") # Non bloquant
        
        self._set_buttons_state(tk.DISABLED)
        self.log_message(f"[GUI] Compression vers {output_path} (Chiffré: {'Oui' if password_for_compression else 'Non'})...")
        thread = threading.Thread(target=self.run_compression, args=(list(self.files_to_compress), output_path, password_for_compression), daemon=True); thread.start()

    def run_compression(self, files_list, output_aic_path, password_compress):
        # --- BLOC TRY-EXCEPT CORRIGÉ ---
        status_msg_for_popup = "Erreur inconnue" # Valeur par défaut
        try:
            success, status_msg_from_core = compress_to_aic(files_list, output_aic_path, 
                                                  password_compress=password_compress, 
                                                  log_callback=self.log_message)
            status_msg_for_popup = status_msg_from_core # Mettre à jour avec le message du core

            if success:
                if self.root.winfo_exists(): 
                    self.root.after(0, lambda path=output_aic_path: messagebox.showinfo("Succès", f"Compression terminée !\nSauvegardé: {path}"))
            else:
                if self.root.winfo_exists(): 
                    self.root.after(0, lambda msg=status_msg_for_popup: messagebox.showerror("Échec", f"Compression échouée.\nMotif: {msg}"))
        except Exception as e:
            self.log_message(f"[GUI] Erreur majeure pendant la compression: {e}")
            import traceback
            self.log_message(f"[GUI] Traceback: {traceback.format_exc()}")
            status_msg_for_popup = str(e) # Utiliser l'exception comme message
            if self.root.winfo_exists(): 
                self.root.after(0, lambda err=e: messagebox.showerror("Erreur Fatale", f"Une erreur inattendue est survenue pendant la compression:\n{err}"))
        finally:
            if self.root.winfo_exists(): 
                self.root.after(0, self._set_buttons_state, tk.NORMAL)
        # --- FIN DU BLOC CORRIGÉ ---

    def start_decompression_thread(self):
        source_archive = self.entry_source_archive_path.get(); dest_folder = self.entry_dest_folder_path.get(); password = self.entry_password_decompress.get()
        if not source_archive or not dest_folder: messagebox.showerror("Erreur", "Archive et destination requises."); return
        if not os.path.exists(source_archive): messagebox.showerror("Erreur", f"Archive '{source_archive}' non trouvée."); return
        self._set_buttons_state(tk.DISABLED)
        self.log_message(f"[GUI] Décompression de {source_archive} (Mdp: {'Oui' if password else 'Non'})...")
        thread = threading.Thread(target=self.run_decompression, args=(source_archive, dest_folder, password if password else None), daemon=True); thread.start()

    def run_decompression(self, archive_path, output_dir, password):
        # --- BLOC TRY-EXCEPT CORRIGÉ ---
        status_code_for_popup = "Erreur inconnue" # Valeur par défaut
        try:
            success, status_code_from_core = extract_archive(archive_path, output_dir, password=password, log_callback=self.log_message)
            status_code_for_popup = status_code_from_core

            if success:
                if self.root.winfo_exists(): 
                    self.root.after(0, lambda path=output_dir: messagebox.showinfo("Succès", f"Décompression terminée !\nFichiers extraits dans: {path}"))
            else:
                password_errors = ["PasswordError", "PasswordNeeded", "BadRarFileOrPassword"] 
                is_pwd_error = any(err_code_part in str(status_code_for_popup) for err_code_part in password_errors) if isinstance(status_code_for_popup, str) else False
                
                if is_pwd_error:
                    self.log_message(f"[GUI] Échec décompression: Mot de passe requis/incorrect pour {archive_path}.")
                    if self.root.winfo_exists(): 
                        self.root.after(0, self.prompt_for_password_and_retry_decompression, archive_path, output_dir)
                        # Ne pas réactiver les boutons ici, prompt_for_password_and_retry_decompression le fera
                        return # Sortir pour éviter le finally global de cette fonction
                else:
                    self.log_message(f"[GUI] Échec décompression '{archive_path}'. Status: {status_code_for_popup}")
                    if self.root.winfo_exists(): 
                        self.root.after(0, lambda code=status_code_for_popup, arc_path=archive_path: messagebox.showerror("Échec", f"Décompression échouée '{os.path.basename(arc_path)}'.\nMotif: {code}"))
        except Exception as e: 
            self.log_message(f"[GUI] Erreur majeure pendant la décompression: {e}"); import traceback
            self.log_message(f"[GUI] Traceback: {traceback.format_exc()}"); 
            status_code_for_popup = str(e)
            if self.root.winfo_exists(): 
                self.root.after(0, lambda err=e: messagebox.showerror("Erreur Fatale", f"Une erreur inattendue est survenue pendant la décompression:\n{err}"))
        finally:
            # Ne réactiver les boutons que si prompt_for_password n'a pas été appelé ou si l'utilisateur a annulé ce prompt
            # prompt_for_password_and_retry_decompression a son propre _set_buttons_state(tk.NORMAL) en cas d'annulation.
            # Si is_pwd_error était True, on est sorti avant. Donc ici, c'est soit succès, soit échec non lié au mdp.
            if self.root.winfo_exists(): 
                # Vérifier si le bouton est toujours désactivé avant de le réactiver
                # Cela évite des appels multiples si prompt_for_password l'a déjà fait
                if self.btn_decompress_action.cget('state') == tk.DISABLED:
                     self.root.after(0, self._set_buttons_state, tk.NORMAL)
        # --- FIN DU BLOC CORRIGÉ ---


    def prompt_for_password_and_retry_decompression(self, archive_path, output_dir):
        if not self.root.winfo_exists(): return
        new_password = simpledialog.askstring("Mot de Passe Requis", 
            f"L'archive '{os.path.basename(archive_path)}' est protégée ou mdp incorrect.\nNouveau mot de passe :", show='*')
        
        if new_password is not None: # Si l'utilisateur entre quelque chose (même une chaîne vide) et clique OK
            self.log_message(f"[GUI] Nouvelle tentative décompression {archive_path} avec mdp fourni.")
            self.entry_password_decompress.delete(0, tk.END)
            self.entry_password_decompress.insert(0, new_password)
            self._set_buttons_state(tk.DISABLED) # Laisser désactivé car on relance
            thread = threading.Thread(target=self.run_decompression, args=(archive_path, output_dir, new_password), daemon=True)
            thread.start()
        else: # L'utilisateur a cliqué sur Annuler
            self.log_message("[GUI] Décompression annulée (pas de nouveau mdp fourni par popup).")
            if self.root.winfo_exists(): 
                self.root.after(0, self._set_buttons_state, tk.NORMAL) # Réactiver les boutons


    def run_check_ota_updates(self): # ... (comme avant)
        if not OTA_AVAILABLE: self.log_message("[GUI] Module OTA non dispo."); messagebox.showwarning("OTA Indisponible", "Module OTA non chargé."); return
        self.log_message("[GUI] Vérification MàJ modèles IA..."); 
        # _set_buttons_state(tk.DISABLED) est appelé dans _perform_ota_check_and_prompt
        # mais on peut le mettre ici pour le bouton spécifique.
        if self.btn_check_ota_updates.winfo_exists(): self.btn_check_ota_updates.config(state=tk.DISABLED, text="Vérification...")
        
        thread = threading.Thread(target=self._perform_ota_check_and_prompt, daemon=True); thread.start()

    def _perform_ota_check_and_prompt(self): # ... (comme avant)
        updates_info_for_gui=[]; models_to_update_actions=[]
        try:
            updates_available = check_for_model_updates(log_callback=self.log_message)
            if not updates_available: updates_info_for_gui.append("Aucune nouvelle MàJ de modèle IA disponible.")
            else:
                updates_info_for_gui.append("MàJ de modèles IA disponibles:"); 
                for model_name, server_info in updates_available.items(): updates_info_for_gui.append(f"  - {model_name}: v{server_info.get('latest_version')} ({server_info.get('description', 'N/A')})"); models_to_update_actions.append((model_name, server_info))
                if models_to_update_actions: updates_info_for_gui.append("\nInstaller ces MàJ ?")
            
            if self.root.winfo_exists(): self.root.after(0, self._show_ota_results_and_prompt_install, updates_info_for_gui, models_to_update_actions)
        except Exception as e: 
            self.log_message(f"[GUI] Erreur vérification OTA: {e}"); import traceback; self.log_message(f"[GUI] Traceback OTA: {traceback.format_exc()}"); 
            if self.root.winfo_exists(): self.root.after(0, lambda err_ota=e: messagebox.showerror("Erreur OTA", f"Erreur MàJ: {err_ota}"))
        finally: 
            # Réactiver seulement si aucune action d'installation n'est proposée, 
            # sinon _show_ota_results ou _perform_model_installations s'en chargent.
            if not models_to_update_actions and self.root.winfo_exists(): 
                 self.root.after(0, lambda: self.btn_check_ota_updates.config(state=tk.NORMAL, text="Vérifier MàJ Modèles IA"))


    def _show_ota_results_and_prompt_install(self, info_messages, models_to_install): # ... (comme avant)
        if not self.root.winfo_exists(): return
        full_message = "\n".join(info_messages); should_ask = bool(models_to_install)
        user_response = False # Par défaut, ne pas installer
        if should_ask:
            # Pour s'assurer que messagebox est appelé dans le thread GUI et qu'on récupère la réponse
            # C'est une partie délicate. Le simpledialog est modal et devrait bloquer.
            # Mais pour être sûr avec after, on pourrait utiliser une variable partagée ou un autre callback.
            # Pour l'instant, on espère que le messagebox modal appelé via after fonctionne.
            user_response = messagebox.askyesno("Mises à Jour Modèles IA", full_message) # Ceci est modal et devrait retourner directement
            
            if user_response:
                self.log_message("[GUI] Installation des MàJ acceptée."); 
                if self.btn_check_ota_updates.winfo_exists(): self.btn_check_ota_updates.config(text="Installation MàJ...") # Pas besoin de _set_buttons_state ici
                thread = threading.Thread(target=self._perform_model_installations, args=(models_to_install,), daemon=True); thread.start()
            else: 
                self.log_message("[GUI] Installation des MàJ refusée.")
                if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, tk.NORMAL) # Réactiver tous
        else: 
            messagebox.showinfo("Mises à Jour Modèles IA", full_message)
            if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, tk.NORMAL) # Réactiver tous


    def _perform_model_installations(self, models_to_install): # ... (comme avant)
        all_successful = True; any_attempted = False
        for model_name, server_info in models_to_install:
            any_attempted = True; self.log_message(f"[GUI] Installation MàJ pour {model_name}...")
            success = download_and_install_model(model_name, server_info, log_callback=self.log_message)
            if success: self.log_message(f"[GUI] MàJ {model_name} réussie !")
            else: self.log_message(f"[GUI] ÉCHEC MàJ {model_name}."); all_successful = False
        
        if self.root.winfo_exists():
            if any_attempted:
                if all_successful: self.root.after(0, lambda: messagebox.showinfo("Mises à Jour IA", "Modèles mis à jour !"))
                else: self.root.after(0, lambda: messagebox.showwarning("Mises à Jour IA", "Certaines MàJ ont échoué."))
        
        if self.root.winfo_exists(): self.root.after(0, self._set_buttons_state, tk.NORMAL)


if __name__ == '__main__':
    app_window = tk.Tk()
    gui_app = AICompressGUI(app_window)
    app_window.mainloop()