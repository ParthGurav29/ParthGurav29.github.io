## Fine-Tuning — Start To End, Simply

---

## Before Anything — What You Need On The PC

```
1. Python installed          → python.org (3.10 or 3.11)
2. CUDA Toolkit installed    → nvidia.com/cuda
3. Git installed             → git-scm.com
4. VS Code installed         → to edit files comfortably
5. RTX 5060 drivers updated  → do this first, critical
```

---

## Phase 1 — Setup The Environment

### Step 1 — Create A Project Folder
```
Make a folder called:
crisisnet-finetune

Put everything inside this folder.
```

### Step 2 — Create A Virtual Environment
```
Open terminal inside that folder

Run:
python -m venv venv

Then activate it:
Windows  →  venv\Scripts\activate
Mac/Linux → source venv/bin/activate

You should see (venv) in your terminal now.
Always work inside this environment.
```

### Step 3 — Install Dependencies
```
Run these one by one in terminal:

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets huggingface_hub

If RTX 5060 gives GPU errors later:
pip install unsloth --pre
```

### Step 4 — Login To Hugging Face
```
Run:
huggingface-cli login

It will ask for a token.
Go to huggingface.co → Settings → Access Tokens
Create a new token → copy it → paste in terminal
Press Enter
```

---

## Phase 2 — Prepare Your Datasets

### Step 5 — Download FirstAidQA From Hugging Face
```
Go to:
huggingface.co/datasets

Search: firstaid qa
Download the dataset files (CSV or JSON format)
Save inside your project folder as:
crisisnet-finetune/data/firstaid_qa.json
```

### Step 6 — Download First Aid Intents From Kaggle
```
Go to kaggle.com
Search: first aid intents dataset
Download the CSV file
Save as:
crisisnet-finetune/data/first_aid_intents.csv
```

### Step 7 — Download MedDialog From Hugging Face
```
Search on Hugging Face:
MedDialog dataset
Download and save as:
crisisnet-finetune/data/meddialog.json
```

---

## Phase 3 — Combine The Datasets

### What Combining Means Simply
```
You have 3 separate files.
You need 1 unified file.
Each entry in the final file looks like:

{
  "input": "Someone is choking",
  "output": "CHOKING — ACT NOW:\n1. Give 5 back blows..."
}

Every single entry from all 3 sources
gets converted into this format
and saved into one file:

crisisnet-finetune/data/combined_dataset.json
```

### How To Do The Combining
```
Step 1 — Open each dataset file
          Look at the column names
          Each dataset has different column names

FirstAidQA might have:    question / answer
MedDialog might have:     description / response  
First Aid Intents might have: text / label

Step 2 — Rename them all to:
          input / output

Step 3 — Copy all entries into one list

Step 4 — Filter out anything not emergency related
          Remove cooking tips, general health questions
          Keep only: first aid, emergencies, disasters, survival

Step 5 — Save as combined_dataset.json

The finetune.py script already does most of this
automatically when you run it.
But you need the files downloaded and in the data/ folder first.
```

### Folder Structure Before Running
```
crisisnet-finetune/
├── venv/
├── data/
│   ├── firstaid_qa.json
│   ├── first_aid_intents.csv
│   └── meddialog.json
├── finetune.py
└── models/              ← empty for now, output goes here
```

---

## Phase 4 — Run The Fine-Tuning

### Step 8 — Place The Script
```
Copy finetune.py into your crisisnet-finetune/ folder
```

### Step 9 — Run It
```
Make sure venv is activated
Then run:

python finetune.py

That's it. Script does everything automatically:
→ Loads datasets
→ Combines them
→ Downloads Gemma 4 E2B from Hugging Face
→ Starts training
→ Shows progress and loss in terminal
→ Tests the model when done
→ Exports GGUF file
```

### What You See In Terminal While Running
```
Loading datasets...
FirstAidQA: loaded 430 examples
MedDialog (filtered): loaded 210 examples
Manual dataset: loaded 19 examples
Total training examples: 659

Loading Gemma 4 E2B with Unsloth...
GPU: NVIDIA RTX 5060
Total VRAM: 8.00 GB

Starting fine-tuning...
Step 10/... | Loss: 1.432
Step 20/... | Loss: 1.287
Step 30/... | Loss: 1.104
...

Training complete.
Time taken: 47.3 minutes
Final loss: 0.421

Running quick test...
Q: Someone stopped breathing
A: RESPIRATORY ARREST — ACT NOW:
   1. Check response...

Exporting to GGUF...
GGUF model ready: ./crisisnet-gguf/model-Q4_K_M.gguf
```

### How Long It Takes
```
Dataset loading    →   2 to 5 minutes
Model loading      →   3 to 5 minutes
Training           →   45 mins to 2 hours (depends on dataset size)
Export             →   10 to 15 minutes

Total: roughly 1 to 2.5 hours
Let it run, don't close the terminal
```

---

## Phase 5 — Get The Output Into Your App

### Step 10 — Find Your GGUF File
```
After script finishes:
crisisnet-finetune/crisisnet-gguf/model-Q4_K_M.gguf

This is your fine-tuned model.
Roughly 2 to 3GB in size.
```

### Step 11 — Replace Model In Your App
```
Go to your React Native project
assets/models/

Delete the old base Gemma GGUF file
Copy model-Q4_K_M.gguf here
Rename it to match whatever name your app expects

Launch the app
Gemma now responds like a survival expert
Done.
```

---

## If Something Goes Wrong

```
CUDA out of memory
→ Open finetune.py
→ Change batch_size from 2 to 1
→ Run again

RTX 5060 not recognized
→ pip install unsloth --pre
→ Run again

Hugging Face dataset not found
→ Search the exact dataset name on huggingface.co
→ Update the dataset name in the script

Model download fails
→ Check your HF token is correct
→ huggingface-cli login again
```

---

## The Entire Process In One View

```
Install Python + CUDA           (once)
       ↓
Create virtual environment      (once)
       ↓
Install dependencies            (once)
       ↓
Login to Hugging Face           (once)
       ↓
Download 3 datasets             (30 mins)
       ↓
Run python finetune.py          (1-2 hours, automatic)
       ↓
Copy GGUF to app                (5 mins)
       ↓
Done 🎯
```