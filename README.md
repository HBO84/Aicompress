# AICompress: Intelligent File Compressor v1.3.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Languages:** **English** | [Français](./README.fr.md)
AICompress is an experimental open-source project that uses an "Orchestrator AI" to intelligently select the most suitable compression method for each file. It aims to provide an optimal balance between compression ratio and speed by leveraging a range of classic algorithms. A key feature is its ability to recursively optimize files within existing archives.

## Key Features

* **Intelligent Compression Selection:** An "Orchestrator AI" (RandomForest model) analyzes files (based on type, size, entropy, quick compressibility) to choose the best algorithm and parameters from its palette.
* **Comprehensive Algorithm Palette:**
    * STORED (no compression)
    * DEFLATE (levels 1, 6, 9 via `zlib`)
    * BZIP2 (level 9 via `bz2`)
    * LZMA (presets 0, 6, 9 via `lzma` and XZ format)
    * Zstandard (levels 1, 3, 9, 15 via `zstandard`)
    * Brotli (qualities 1, 6, 11 via `brotli`)
* **Recursive Archive Optimization:** When compressing, AICompress can detect inner archives (ZIP, RAR, 7z, AIC) within your selection. If enabled, it extracts these archives and intelligently recompresses their contents individually into the final `.aic` archive.
* **Custom Archive Format (`.aic`):** Based on the standard ZIP format, containing processed files and an internal `aicompress_metadata.json` metadata file detailing the operations performed on each file.
* **Encryption:** Optional password protection for `.aic` archives using AES-256 in GCM mode.
* **Multi-format Decompression:**
    * `.aic` archives created by the application.
    * Standard `.zip` archives.
    * `.rar` archives (requires external `unrar` tool).
    * `.7z` archives (via the `py7zr` Python library).
* **Graphical User Interface (GUI):** Developed with Tkinter, featuring:
    * File and folder selection for compression.
    * Output file and destination folder selection.
    * Password encryption option.
    * Detailed progress bars for compression and decompression (per-file for AIC, ZIP, RAR; indeterminate for 7z).
    * "Cancel" buttons for long operations.
    * Operation logs display.
* **OTA (Over-The-Air) Updates for AI Models:** Basic infrastructure is in place (functionality to be fully tested/finalized).

## Project Status

**Version 1.3.0 - Alpha.**
This project is under active development and is primarily a field for experimentation and learning. Bugs may be present, and changes may occur.

## Installation

1.  **Prerequisites:**
    * Python 3.10 - 3.12 recommended (developed and tested with 3.12).
    * `pip` (Python package manager).
    * For `.rar` decompression, the `unrar` command-line tool must be installed on your system and accessible in your system's PATH.
    * On some Linux systems, for file type detection with `python-magic`, installing `libmagic1` (or a similar package) may be necessary:
        ```bash
        sudo apt-get update && sudo apt-get install libmagic1
        ```
    * For Windows, if running from source, `python-magic-bin` is recommended to provide necessary DLLs for `python-magic`.

2.  **Clone the Repository (if public):**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/AICompressProject.git](https://github.com/YOUR_USERNAME/AICompressProject.git)
    cd AICompressProject
    ```

3.  **Create a Virtual Environment (Highly Recommended):**
    ```bash
    python3 -m venv aic_env
    source aic_env/bin/activate  # On Linux/macOS
    # or aic_env\Scripts\activate   # On Windows
    ```

4.  **Install Dependencies:**
    At the root of the project, you will find a `requirements.txt` file. Install the dependencies with:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Activate your virtual environment:
    ```bash
    source aic_env/bin/activate # or Windows equivalent
    ```
2.  Run the GUI from the project root:
    ```bash
    python aicompress_gui.py
    ```
3.  Use the interface to:
    * Add files or folders to the compression list.
    * Optionally, check "Optimiser les archives incluses" to enable recursive optimization of archives within your selection.
    * Specify the output `.aic` archive name and location.
    * Choose to enable password encryption.
    * Start compression.
    * Select an archive (`.aic`, `.zip`, `.rar`, `.7z`) and a destination folder for decompression.

## How the AI Works (Overview)

* **AI Analyzer (`aicompress/ai_analyzer.py`):** Determines the file type (text, image, binary, Python script, etc.) and extracts several numerical features: size, normalized entropy, and a "quick compressibility ratio."
* **Orchestrator AI (`aicompress/orchestrator.py`):** A Machine Learning model (RandomForest) is trained on a pre-computed dataset. It uses the features extracted by the analyzer to predict the "best" compression method (and its level/preset/quality) from the available classic algorithms (STORED, DEFLATE, BZIP2, LZMA, Zstandard, Brotli).

## Roadmap / Future Features

* Continuous improvement of the "Orchestrator AI" accuracy and decision-making.
* GUI enhancements: Stable window resizing, drag-and-drop functionality.
* Refining progress bar for recursive operations (e.g., more accurate total file count).
* Finalizing and testing OTA updates for AI models.
* Advanced packaging for easier distribution on Windows, Linux, and macOS.
* Exploring further performance optimizations (e.g., multiprocessing for `create_decision_dataset.py`).

## Contributing


This project is currently a personal development. Suggestions and feedback are welcome via GitHub Issues.

## License

This project is distributed under the terms of the **MIT License**.
See the `LICENSE` file at the root of the project for more details.