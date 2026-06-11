# How to Run the Compliance Validator on AMD Developer Cloud (Jupyter)

Follow these steps to set up and run the AI-Driven Audit & Compliance Validator on your AMD Developer Cloud instance.

---

## Step 1: Clone the Project or Fetch Files
In your Jupyter Notebook, open a new cell and download your project files from your Git repository. 

Run this command inside a cell:
```bash
!git clone <YOUR_GITHUB_REPOSITORY_URL>
%cd <REPOSITORY_NAME>
```

*Alternatively, if you only want to download the files individually since file upload is disabled:*
```python
import urllib.request

# Download compliance_agent.py
urllib.request.urlretrieve("https://raw.githubusercontent.com/<user>/<repo>/main/compliance_agent.py", "compliance_agent.py")
# Download generate_samples.py
urllib.request.urlretrieve("https://raw.githubusercontent.com/<user>/<repo>/main/generate_samples.py", "generate_samples.py")
# Download requirements.txt
urllib.request.urlretrieve("https://raw.githubusercontent.com/<user>/<repo>/main/requirements.txt", "requirements.txt")
```

---

## Step 2: Install the Dependencies
Run this cell in Jupyter to install the necessary libraries on the notebook instance:
```bash
!pip install -r requirements.txt
```
*Note: AMD Developer Cloud containers come with `torch` (ROCm-optimized) pre-installed. Pip will satisfy this requirement automatically without replacing the ROCm version.*

---

## Step 3: Verify the AMD GPU (ROCm)
Confirm PyTorch detects your AMD GPU (such as Instinct MI210 or MI300) correctly. Paste this code into a cell:
```python
import torch
print("ROCm/CUDA GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("ROCm Version:", torch.version.hip if hasattr(torch.version, 'hip') else "Standard CUDA")
else:
    print("WARNING: Running on CPU. Verify your AMD PyTorch kernel is active.")
```

---

## Step 4: Generate Testing Documents
Run the sample generator script. This will programmatically create the compliance rules and a sample mortgage contract with intentional violations:
```python
!python generate_samples.py
```
This creates:
- `sample_data/rules.json` (List of compliance rules)
- `sample_data/sample_contract.txt` (A mortgage contract text file)

---

## How to Input Your Custom Local Policies (PDF / TXT)

If you have a custom policy file saved locally on the container (e.g., `my_policy.txt` or `my_policy.pdf`), you can load it into the notebook as follows:

### 1. For a Text File (.txt)
```python
doc_path = "my_policy.txt"
with open(doc_path, "r", encoding="utf-8") as f:
    document_text = f.read()
```

### 2. For a PDF File (.pdf)
```python
import pypdf

doc_path = "my_policy.pdf"
reader = pypdf.PdfReader(doc_path)
document_text = ""
for page in reader.pages:
    document_text += page.extract_text() + "\n"
```

---

## Step 5: Configure and Run the Agentic Audit
Open the **`compliance_validator.ipynb`** notebook and locate the execution cells:

1. **Load Data**: Load the rule set and contract:
   ```python
   import json
   with open("sample_data/rules.json", "r") as f:
       rules = json.load(f)
   with open("sample_data/sample_contract.txt", "r") as f:
       document_text = f.read()
   ```

2. **Select LLM Execution Mode**:
   - For **Full Offline GPU Execution**, set `MODEL_TYPE = "hf"` and select a lightweight model like Qwen:
     ```python
     MODEL_TYPE = "hf"
     MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Downloaded and run locally on ROCm
     API_KEY = ""
     ```
   - For **Quick API Validation**, set `MODEL_TYPE = "gemini"` and provide your key:
     ```python
     MODEL_TYPE = "gemini"
     MODEL_NAME = "gemini-1.5-flash"
     API_KEY = "AIzaSy..."  # Your Google API Key
     ```
   - For **Deterministic Mock Verification** (no downloads or APIs):
     ```python
     MODEL_TYPE = "mock"
     ```

3. **Run the Audit**:
   ```python
   from compliance_agent import run_compliance_audit
   audit_reports = run_compliance_audit(
       document_text=document_text,
       rules=rules,
       model_type=MODEL_TYPE,
       model_name=MODEL_NAME,
       api_key=API_KEY
   )
   ```

---

## Step 6: View the Visual Dashboard
Run the reporting cells in the notebook to print out the compliance status:
- **Pandas Summary Table**: Renders a color-coded table (PASS: green, FAIL: red, WARN: orange).
- **Audit Cards**: Displays formatted HTML blocks containing the exact retrieved sentences as evidence and validation checks verifying the lack of AI hallucinations.
- **Export files**: Automatically creates `audit_report.json` and `audit_report.md` on the container disk.
