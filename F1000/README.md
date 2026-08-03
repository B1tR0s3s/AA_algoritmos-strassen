## Repository Execution
### 1. F1000 - MCM + Strassen Notebook
#### Measurement Execution

#### In Visual Studio Code
Requirements:
- Python 3
- Jupyter extension for Visual Studio Code or access to Google Colab
- Google account (if Google Colab will be used)

Instructions:
- Open the notebook in Visual Studio Code.
- Select a compatible runtime: local Jupyter or Google Colab.
- If using Google Colab, sign in with a Google account.
- Verify that the Python 3 kernel is selected.
- Run all notebook cells using the Run All option.
- Wait for execution to finish to obtain the corresponding measurements and results.

The original notebook was used as the base. To obtain real results by matrix size, the static data cell was replaced with an execution cell that generates matrices, computes standard and Strassen multiplication, and measures time, memory, and numerical error.


#### Metrics Obtained:

- Total execution time
- Peak RAM usage
- Average RAM usage
- Time per cell
