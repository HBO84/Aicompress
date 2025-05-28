# AICompress: Intelligent File Compressor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Languages:** **English** | [Français](./README.md)
AICompress is an experimental file compression application that uses an artificial intelligence approach (an "Orchestrator AI") to choose the most suitable compression method for each file type. It aims to provide a good balance between compression ratio and speed, while also exploring the use of neural techniques for specific use cases.

## Current Features

* **Intelligent Compression:** An "Orchestrator AI" (RandomForest model) analyzes files and selects from several classic compression algorithms.
* **AI-Supported Algorithms:**
    * STORED (no compression)
    * DEFLATE (levels 1, 6, 9 via `zlib`)
    * BZIP2 (level 9 via `bz2`)
    * LZMA (presets 0, 6, 9 via `lzma` and XZ format)
    * Zstandard (levels 1, 3, 9, 15 via `zstandard`)
    * Brotli (qualities 1, 6, 11 via `brotli`)
* **Experimental Autoencoder Engine:** A neural autoencoder (Keras/TensorFlow) for lossy compression of small/medium-sized color images (currently suited for 32x32 pixel CIFAR-10 like images).
* **Custom Archive Format `.aic`:** Based on the standard ZIP format, containing processed files and an internal `aicompress_metadata.json` metadata file.
* **Encryption:** Optional password protection for `.aic` archives using AES-256 in GCM mode.
* **Multi-format Decompression:**
    * `.aic` archives created by the application.
    * Standard `.zip` archives.
    * `.rar` archives (requires external `unrar` tool).
    * `.7z` archives (via the `py7zr` Python library).
* **Graphical User Interface (GUI):** Developed with Tkinter.
    * Add files and folders.
    * Select output file and destination folder.
    * Password encryption option.
    * Detailed progress bars for compression and decompression.
    * "Cancel" buttons for long operations.
    * Operation logs display.
* **OTA (Over-The-Air) Updates:** Basic infrastructure for updating AI models (feature to be fully tested and finalized).

## Project Status

**Alpha Version.**
This project is under active development and is primarily a field for experimentation and learning. Bugs may be present, and major changes may occur.

## Installation

1.  **Prerequisites:**
    * Python 3.12 recommended.
    * `pip` (Python package manager).
    * For `.rar` decompression, the `unrar` command-line tool must be installed on your system and accessible in your system's PATH.
    * On some Linux systems, for file type detection with `python-magic`, installing `libmagic1` (or a similar package) may be necessary:
        ```bash
        sudo apt-get update && sudo apt-get install libmagic1
        ```

2.  **Clone the Repository (if public):**
    ```bash
    git clone https://github.com/hbo84/AICompress.git
    cd AICompress
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
    * Specify the output `.aic` archive name and location.
    * Choose to enable password encryption.
    * Start compression.
    * Select an archive (`.aic`, `.zip`, `.rar`, `.7z`) and a destination folder for decompression.

## How the AI Works (Overview)

* **AI Analyzer (`aicompress/ai_analyzer.py`):** Determines the file type (text, image, binary, Python script, etc.) and extracts several numerical features: size, normalized entropy, and a "quick compressibility ratio."
* **Orchestrator AI (`aicompress/orchestrator.py`):** A Machine Learning model (RandomForest) trained on a pre-computed dataset. It uses the features extracted by the analyzer to predict the "best" compression method (and its level/preset) from the available options (STORED, DEFLATE, BZIP2, LZMA, Zstandard, Brotli, AE Engine).
* **Autoencoder Engine (`aicompress/ae_engine.py`):** A Keras/TensorFlow convolutional autoencoder model used for lossy compression of small color images. The Orchestrator AI can decide to use this engine for compatible images.

## Roadmap / Future Features (Examples)

* Continuous improvement of the "Orchestrator AI" accuracy and relevance.
* Recursive handling of input archives (decompress then intelligently recompress).
* GUI enhancements: stable window resizing, drag-and-drop.
* Finalizing and testing OTA updates for AI models.
* Packaging the application for easier distribution (Windows/Linux executables).
* Exploring new neural compression engines (lossy or lossless).

## Contributing

This project is currently a personal development. Suggestions and feedback are welcome via GitHub Issues.

## License

This project is distributed under the terms of the **MIT License**.
See the `LICENSE` file at the root of the project for more details.