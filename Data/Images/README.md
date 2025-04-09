
# Downloading and Unzipping the Dataset File

This guide provides command-line instructions to download and unzip the dataset file available at the specified URL using common tools available on Windows, Linux, and macOS.

**File URL:** `https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip`
**Expected Downloaded Filename:** `ptz377bwb8-1.zip`

## Instructions by Operating System

Choose the section corresponding to your operating system. It's recommended to navigate to your preferred download directory first (e.g., your Downloads folder).

---

### Windows (Command Prompt - `cmd`)

1.  **Open Command Prompt:**
    * Press `Win + R`, type `cmd`, and press Enter.

2.  **Navigate to Download Folder (Recommended):**
    ```cmd
    cd %UserProfile%\Downloads
    ```
    *(Adjust the path `%UserProfile%\Downloads` if you prefer a different location)*

3.  **Download the file using `curl`:**
    ```cmd
    curl -L -O [https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip)
    ```
    * `curl`: Tool to transfer data.
    * `-L`: Follow server redirects.
    * `-O`: Save file using the name from the URL.

4.  **Unzip the file using `tar`:**
    *(Requires a modern version of Windows 10 or Windows 11)*
    ```cmd
    tar -xf ptz377bwb8-1.zip
    ```
    * `tar`: Archive tool included in modern Windows.
    * `-x`: Extract files.
    * `-f`: Specify the archive file name.
    *(If `tar -xf` fails, see "Important Notes" below).*

---

### Linux (Bash / Standard Terminal)

1.  **Open your Terminal.**

2.  **Navigate to Download Folder (Recommended):**
    ```bash
    cd ~/Downloads
    ```
    *(Adjust `~/Downloads` if you prefer a different location)*

3.  **Download the file (Choose one option):**

    * **Option A: Using `curl`**
        ```bash
        curl -L -O [https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip)
        ```

    * **Option B: Using `wget`**
        ```bash
        wget [https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip)
        ```

4.  **Unzip the file using `unzip`:**
    ```bash
    unzip ptz377bwb8-1.zip
    ```
    * `unzip`: Standard utility for uncompressing `.zip` files.
    *(If `unzip` is not found, install it using your distribution's package manager, e.g., `sudo apt update && sudo apt install unzip` for Debian/Ubuntu or `sudo yum install unzip` for CentOS/Fedora).*

---

### macOS (Terminal)

1.  **Open Terminal:**
    * Go to `Applications` -> `Utilities` -> `Terminal`.

2.  **Navigate to Download Folder (Recommended):**
    ```bash
    cd ~/Downloads
    ```
    *(Adjust `~/Downloads` if you prefer a different location)*

3.  **Download the file using `curl`:**
    ```bash
    curl -L -O [https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip](https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/ptz377bwb8-1.zip)
    ```

4.  **Unzip the file using `unzip`:**
    ```bash
    unzip ptz377bwb8-1.zip
    ```
    * `unzip`: Standard utility, typically pre-installed on macOS.

---

## Important Notes

* **Permissions:** Ensure you have permission to write files in your chosen download and extraction directory.
* **Disk Space:** Make sure you have enough free disk space for both the downloaded `.zip` file and the files contained within it once extracted.
* **Network:** A stable internet connection is required for the download. If you are behind a corporate proxy, `curl` or `wget` might require additional configuration flags (e.g., `-x proxy_server:port`).
* **Windows `tar` Alternative:** If the `tar -xf` command fails on Windows (likely due to an older version), you can try using PowerShell's `Expand-Archive` command directly from `cmd`:
    ```cmd
    powershell -Command "Expand-Archive -Path 'ptz377bwb8-1.zip' -DestinationPath '.'"
    ```
    Alternatively, install a third-party tool like [7-Zip](https://www.7-zip.org/) which provides a command-line interface.
* **Extraction Location:** The `unzip` and `tar -xf` commands shown will extract the contents directly into the current directory where the zip file resides.
