# 🩺 AI Physician Notetaker

This project provides a sophisticated AI pipeline for processing and analyzing medical conversation transcripts. It automatically extracts a structured medical summary, analyzes patient sentiment and intent, and generates a clinically relevant SOAP note.

The system is built using a powerful hybrid approach, combining:

  * **Rule-Based Extraction** for high-precision clinical data extraction.
  * **Transformer-based Zero-Shot Classification** for fast and flexible sentiment analysis.
  * **A Large Language Model (LLM)** for dynamic and context-aware SOAP note generation.

-----

## \#\# Features

  * **Automated Medical Summary:** Extracts key details like symptoms, diagnosis, and treatments into a structured JSON format.
  * **Patient Sentiment Analysis:** Identifies the patient's emotional state (e.g., `Anxious`, `Reassured`) and intent (e.g., `Reporting symptoms`).
  * **Dynamic SOAP Note Generation:** Converts the raw conversation into a professional, structured SOAP (Subjective, Objective, Assessment, Plan) note.
  * **Unified Pipeline:** A single, easy-to-use class that runs the complete analysis with one command.

-----

## \#\# Setup and Installation

This project is designed to be run in a Python environment, such as a local machine or a cloud-based notebook like Google Colab.

### \#\#\# 1. Prerequisites

  * **Python 3.8+**
  * **pip** (Python package installer)

### \#\#\# 2. Create a Virtual Environment (Recommended)

To avoid conflicts with other projects, it's best to create a dedicated virtual environment.

```bash
# Create the virtual environment
python -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# Or activate it (on Windows)
.\venv\Scripts\activate
```

### \#\#\# 3. Install Dependencies

Install all the required Python libraries using the following command:

```bash
pip install torch transformers sentence-transformers
```

**Note:** `torch` will be installed as a dependency. For faster performance on a machine with a compatible NVIDIA GPU, you can install a CUDA-enabled version of PyTorch by following the instructions on the official [PyTorch website](https://pytorch.org/).

-----

## \#\# How to Run the Pipeline

1.  **Save the Code:** Save the final Python script as a file (e.g., `medical_pipeline.py`).

2.  **Provide Your Input:** Open the file and locate the `if __name__ == "__main__":` block at the bottom. Modify the `user_transcript` variable to include the conversation you want to analyze.

    ```python
    if __name__ == "__main__":
        # ...
        # --- Step 2: Provide your user-defined input transcript HERE ---
        user_transcript = """
        Patient: Good afternoon, doctor. I've had this sharp pain in my knee...
        ...
        """
        # ...
    ```

3.  **Execute the Script:** Run the script from your terminal.

    ```bash
    python medical_pipeline.py
    ```

The script will automatically download the necessary AI models from the Hugging Face Hub on its first run. These models will be cached locally for all subsequent runs. The final, structured JSON output containing all three analyses will be printed directly to your console.
