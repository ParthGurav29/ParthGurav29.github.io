# ============================================================
# CrisisNet — Gemma 4 E2B Fine-Tuning Script
# GPU: RTX 5060 (Blackwell) or any RTX with 8GB+ VRAM
# Framework: Unsloth + HuggingFace Transformers
# ============================================================

# ── STEP 0 — INSTALL DEPENDENCIES ───────────────────────────
# Run this once in terminal before running the script:
#
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps trl peft accelerate bitsandbytes
# pip install datasets huggingface_hub
#
# For RTX 5060 (Blackwell arch), if unsloth has issues:
# pip install unsloth --pre  (pre-release supports newer GPUs)
# ─────────────────────────────────────────────────────────────

import os
import json
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ============================================================
# STEP 1 — CONFIGURATION
# Tweak these values based on your GPU memory
# ============================================================

CONFIG = {
    # Model
    "base_model"     : "unsloth/gemma-4-E2B-it",   # base model from Hugging Face
    "max_seq_length" : 1024,                         # max tokens per example
    "load_in_4bit"   : True,                         # 4-bit quantization saves VRAM

    # LoRA (Low-Rank Adaptation) settings
    # These control how much of the model gets fine-tuned
    "lora_r"         : 32,    # Increased to 32 for better capture of critical nuances
    "lora_alpha"     : 64,    # scaling factor, usually 2x lora_r
    "lora_dropout"   : 0.05,  # regularization, small value is fine

    # Training settings
    "batch_size"            : 2,      # reduce to 1 if you get CUDA out of memory
    "gradient_accumulation" : 4,      # effective batch = batch_size x this
    "epochs"                : 3,      # 2-3 is enough, more risks overfitting
    "learning_rate"         : 2e-4,   # standard for LoRA fine-tuning
    "warmup_steps"          : 10,     # gradual LR warmup at start
    "lr_scheduler"          : "cosine",         # Cosine decay for better convergence
    "optimizer"             : "paged_adamw_8bit", # Memory efficient optimizer
    "weight_decay"          : 0.01,             # Prevent overfitting to training data

    # Output
    "output_dir"            : "./crisisnet-model",         # where checkpoints save
    "final_model_dir"       : "./crisisnet-gemma-final",   # final merged model
    "gguf_output_dir"       : "./crisisnet-gguf",          # GGUF for mobile
    "gguf_quantization"     : "q4_k_m",                    # mobile-ready quant
}


# ============================================================
# STEP 2 — DATASET PREPARATION
# Combines multiple sources into one unified dataset
# ============================================================

# ── Prompt template ─────────────────────────────────────────
# This is how every training example gets formatted.
# Gemma uses a specific chat format — this follows it.

SYSTEM_PROMPT = """You are CrisisNet AI — a critical emergency and survival assistant.
Your sole purpose is to provide immediate, life-saving instructions.
RULES:
1. Respond ONLY with SHORT, NUMBERED, ACTIONABLE steps.
2. Start with the emergency name in CAPS.
3. Use direct, imperative commands (e.g., "Do this", NOT "You should do this").
4. List the most critical, life-saving action as Step 1.
5. Use zero medical jargon. Keep language simple and calm.
6. Always end by instructing to call emergency services if not already stated.
Do not include any pleasantries or conversational filler."""

def format_prompt(input_text, output_text):
    """Formats a QA pair into Gemma chat format for training."""
    return f"""<start_of_turn>system
{SYSTEM_PROMPT}<end_of_turn>
<start_of_turn>user
{input_text}<end_of_turn>
<start_of_turn>model
{output_text}<end_of_turn>"""


# ── Load and merge datasets ──────────────────────────────────

def load_firstaid_qa():
    """Load FirstAidQA dataset from Hugging Face."""
    print("Loading FirstAidQA dataset...")
    try:
        # Try loading the dataset — update dataset name if different on HF
        ds = load_dataset("aszarata/firstaid-qa", split="train")
        examples = []
        for row in ds:
            # Adjust column names based on actual dataset structure
            question = row.get("question") or row.get("input") or row.get("prompt", "")
            answer   = row.get("answer")   or row.get("output") or row.get("response", "")
            if question and answer:
                examples.append({
                    "text": format_prompt(question, answer)
                })
        print(f"FirstAidQA: loaded {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"FirstAidQA load failed: {e}. Skipping.")
        return []


def load_meddialog_emergency():
    """Load MedDialog and filter only emergency-related conversations."""
    print("Loading MedDialog dataset (emergency filter)...")

    # Keywords that signal emergency-relevant conversations
    EMERGENCY_KEYWORDS = [
        "emergency", "urgent", "immediately", "heart attack", "stroke",
        "bleeding", "unconscious", "breathing", "choking", "fracture",
        "burn", "poison", "overdose", "drowning", "seizure", "shock",
        "allergic", "accident", "injury", "wound", "collapse"
    ]

    try:
        ds = load_dataset("super_glue", "meddialog", split="train")
        examples = []
        for row in ds:
            text = str(row).lower()
            if any(keyword in text for keyword in EMERGENCY_KEYWORDS):
                question = row.get("description") or row.get("input", "")
                answer   = row.get("utterances",  [{}])
                if isinstance(answer, list) and len(answer) > 0:
                    answer = answer[-1].get("utterance", "")
                if question and answer:
                    examples.append({
                        "text": format_prompt(str(question), str(answer))
                    })
        print(f"MedDialog (filtered): loaded {len(examples)} examples")
        return examples
    except Exception as e:
        print(f"MedDialog load failed: {e}. Skipping.")
        return []


def load_manual_survival_dataset():
    """
    Core hand-crafted survival dataset.
    These are the most important entries — verified against
    Red Cross, WHO, and FEMA guidelines.
    Add more entries following the same format.
    """
    print("Loading manual survival dataset...")
    entries = [
        # ── MEDICAL ─────────────────────────────────────────
        {
            "input": "Someone near me stopped breathing",
            "output": """RESPIRATORY ARREST — ACT NOW:
1. Check response — tap shoulders, shout their name
2. Call emergency services immediately
3. Tilt head back, lift chin to open airway
4. Give 2 rescue breaths — watch chest rise
5. Begin 30 chest compressions — hard and fast
6. Repeat 30 compressions + 2 breaths until help arrives
Do NOT stop until a medic takes over."""
        },
        {
            "input": "How do I do CPR?",
            "output": """CPR — ACT NOW:
1. Lay person flat on their back
2. Place heel of hand on center of chest
3. Push down hard and fast — 30 compressions
   Rate: 100-120 per minute (beat of Stayin Alive)
   Depth: at least 2 inches
4. Give 2 rescue breaths — tilt head, seal mouth, blow
5. Repeat: 30 compressions, 2 breaths
6. Continue until they breathe or help arrives
Call emergency services before starting if alone."""
        },
        {
            "input": "Someone is choking and cannot speak",
            "output": """CHOKING — ACT NOW:
1. Confirm they cannot speak, cough, or breathe
2. Lean them forward, support their chest
3. Give 5 firm back blows between shoulder blades
4. Give 5 abdominal thrusts — fist above navel, pull inward-upward
5. Alternate 5 back blows + 5 thrusts until object clears
6. If unconscious — begin CPR immediately
Call emergency services now."""
        },
        {
            "input": "There is severe bleeding that won't stop",
            "output": """SEVERE BLEEDING — ACT NOW:
1. Apply firm pressure with cloth or clothing
2. Press HARD and do NOT remove — blood soaking through means press harder on top
3. If limb — raise it above heart level
4. Pack wound tightly if deep
5. Tie tourniquet above wound ONLY if bleeding is life-threatening
6. Keep them warm and lying down — prevents shock
7. Do not remove the cloth — keep pressing
Call emergency services immediately."""
        },
        {
            "input": "Someone is having a heart attack",
            "output": """HEART ATTACK — ACT NOW:
1. Call emergency services immediately
2. Sit them down — do not let them walk
3. Loosen tight clothing around neck and chest
4. Give aspirin (300mg) if available and not allergic — chew it
5. Keep them calm and still
6. If unconscious and not breathing — begin CPR
Do NOT leave them alone."""
        },
        {
            "input": "I think someone is having a stroke",
            "output": """STROKE — USE FAST METHOD:
F — Face: Ask them to smile. Is one side drooping?
A — Arms: Ask them to raise both. Does one drift down?
S — Speech: Ask them to repeat a phrase. Is it slurred?
T — Time: If ANY of the above — call emergency NOW

DO NOT:
- Give food or water
- Let them sleep it off
- Drive them yourself if ambulance is available
Every minute matters — call immediately."""
        },
        {
            "input": "Someone has a severe burn",
            "output": """BURN — ACT NOW:
1. Remove from heat source — stop the burning
2. Cool burn under cool running water for 20 minutes minimum
3. Remove clothing and jewelry NEAR the burn — not stuck to it
4. Cover loosely with clean cling film or non-fluffy cloth
5. Do NOT use ice, butter, or toothpaste — worsens damage
6. Do NOT burst blisters
7. Seek emergency care for burns larger than palm size
Call emergency services for face, hands, or deep burns."""
        },
        {
            "input": "Someone is in shock",
            "output": """SHOCK — ACT NOW:
Signs: pale/cold skin, rapid weak pulse, confusion, dizziness
1. Lay them flat on their back
2. Raise their legs above heart level — unless head/spine injury
3. Keep them warm — cover with blanket
4. Do NOT give food or water
5. Loosen tight clothing
6. Talk to them — keep them conscious and calm
7. Monitor breathing constantly
Call emergency services immediately."""
        },
        {
            "input": "Someone broke a bone or has a fracture",
            "output": """FRACTURE — ACT NOW:
1. Do NOT move them if spine or pelvis may be injured
2. Keep the injured area still — do not straighten it
3. Support above and below the fracture
4. Apply ice wrapped in cloth — reduces swelling
5. Improvise splint with rigid object and padding if moving is necessary
6. Check circulation below injury — is skin warm? Can they feel?
7. Treat for shock if needed
Seek emergency care for all suspected fractures."""
        },
        {
            "input": "Someone is having a seizure",
            "output": """SEIZURE — ACT NOW:
1. Clear area — remove sharp or hard objects nearby
2. Cushion their head with something soft
3. Do NOT hold them down or restrain movement
4. Do NOT put anything in their mouth
5. Time the seizure — note when it started
6. After seizure — roll them onto their side (recovery position)
7. Stay with them until fully conscious

Call emergency services if:
- Seizure lasts more than 5 minutes
- Second seizure follows immediately
- They do not regain consciousness
- This is their first seizure"""
        },

        # ── DISASTER ────────────────────────────────────────
        {
            "input": "Earthquake is happening right now",
            "output": """EARTHQUAKE — ACT NOW:
IF INDOORS:
1. DROP to hands and knees immediately
2. COVER — get under sturdy desk/table or against interior wall
3. HOLD ON until shaking stops
4. Stay away from windows, exterior walls, heavy furniture
5. Do NOT run outside during shaking

WHEN SHAKING STOPS:
1. Check for injuries — treat before moving
2. Expect aftershocks
3. Do NOT use elevator
4. Exit building carefully — watch for falling debris
5. Move to open ground away from buildings"""
        },
        {
            "input": "I am trapped in a collapsed building",
            "output": """TRAPPED IN COLLAPSE — DO THIS:
1. Stay calm — panic wastes oxygen and energy
2. Cover mouth with clothing — filter dust
3. Do NOT light matches or lighters — gas leaks possible
4. Signal rescuers — tap on pipes or walls 3 times repeatedly
5. Shout only when you hear rescuers — saves energy
6. Conserve phone battery — send location once then wait
7. Do NOT move debris above you — may cause further collapse
8. If water is rising — move to highest point possible

You will be found. Stay still. Keep signaling."""
        },
        {
            "input": "There is a flash flood happening",
            "output": """FLASH FLOOD — ACT NOW:
1. Move to high ground IMMEDIATELY — do not wait
2. Do NOT walk through moving water — 6 inches can knock you down
3. Do NOT drive through flooded roads — turn around
4. Stay away from streams, drainage channels, bridges
5. If swept away — float on your back, feet first, downstream
6. Grab onto something stable — tree, pole, solid structure
7. Signal for rescue from high ground

Do NOT return until authorities confirm it is safe."""
        },
        {
            "input": "There is a fire in my building",
            "output": """BUILDING FIRE — ACT NOW:
1. Activate nearest fire alarm if not already triggered
2. Call emergency services immediately
3. Feel door before opening — if hot, do NOT open
4. If safe to exit — stay LOW, crawl under smoke
5. Close doors behind you — slows fire spread
6. Use stairs ONLY — never elevator
7. If blocked by fire — seal door gaps with cloth
8. Signal from window — do not jump unless last resort
9. Once outside — do NOT go back in for any reason

Smoke kills faster than fire — stay low always."""
        },

        # ── SURVIVAL ────────────────────────────────────────
        {
            "input": "I am lost and don't know where I am",
            "output": """LOST — STOP METHOD:
S — STOP: Sit down. Calm your breathing. Do not panic.
T — THINK: When did you last know your location? What direction?
O — OBSERVE: Landmarks, sun position, water flow direction
P — PLAN: Choose ONE direction and move consistently

SIGNAL FOR RESCUE:
- Stay in open areas — easier to spot from air
- 3 signals = universal distress (3 fires, 3 whistle blasts)
- Use mirror or shiny object to reflect sunlight
- Write SOS in large letters on open ground

Moving randomly makes you harder to find. Stay calm."""
        },
        {
            "input": "How do I find safe drinking water in an emergency?",
            "output": """EMERGENCY WATER — IN ORDER OF SAFETY:
1. Stored bottled water — use this first
2. Rainwater — collect in clean containers
3. Moving water — streams and rivers (must purify)
4. Still water — lakes and ponds (must purify)
5. NEVER drink seawater or flood water

TO PURIFY:
- Boil for 1 full minute (3 min above 6500ft altitude)
- Use water purification tablets if available
- Filter through cloth first to remove debris

Signs of dehydration: dark urine, dizziness, dry mouth
Drink at least 2 liters per day minimum."""
        },
        {
            "input": "Someone is showing signs of heatstroke",
            "output": """HEATSTROKE — ACT NOW:
Signs: body temp above 104F, no sweating, confusion, hot dry skin
1. Move to shade or cool area immediately
2. Remove excess clothing
3. Cool them rapidly — wet cloth on neck, armpits, groin
4. Fan them continuously
5. Give cool water ONLY if they are conscious and can swallow
6. Do NOT give fever medication — does not help heatstroke
7. Place in recovery position if unconscious
Call emergency services — heatstroke is life-threatening."""
        },
        {
            "input": "Someone is hypothermic — very cold and confused",
            "output": """HYPOTHERMIA — ACT NOW:
Signs: intense shivering, confusion, slurred speech, drowsiness
1. Move to warm dry shelter immediately
2. Remove wet clothing — replace with dry layers
3. Warm the core first — chest, neck, groin, armpits
4. Use blankets, body heat, warm water bottles wrapped in cloth
5. Give warm sweet drinks ONLY if conscious and alert
6. Do NOT rub limbs — drives cold blood to heart
7. Do NOT give alcohol
8. Handle gently — cold heart is prone to stopping
Call emergency services for severe hypothermia."""
        },
    ]

    formatted = [
        {"text": format_prompt(e["input"], e["output"])}
        for e in entries
    ]
    print(f"Manual dataset: loaded {len(formatted)} examples")
    return formatted


def build_combined_dataset():
    """Combine all sources into one training dataset."""
    all_examples = []

    all_examples += load_manual_survival_dataset()   # highest priority
    all_examples += load_firstaid_qa()               # from Hugging Face
    all_examples += load_meddialog_emergency()        # filtered emergency only

    print(f"\nTotal training examples: {len(all_examples)}")

    # Convert to HuggingFace Dataset format
    return Dataset.from_list(all_examples)


# ============================================================
# STEP 3 — LOAD MODEL WITH UNSLOTH
# ============================================================

def load_model():
    print("\nLoading Gemma 4 E2B with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = CONFIG["base_model"],
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,          # auto-detect best dtype for your GPU
        load_in_4bit   = CONFIG["load_in_4bit"],
    )

    # Apply LoRA adapters — this is what gets trained
    # Only a small % of model weights get updated — very efficient
    model = FastLanguageModel.get_peft_model(
        model,
        r              = CONFIG["lora_r"],
        lora_alpha     = CONFIG["lora_alpha"],
        lora_dropout   = CONFIG["lora_dropout"],
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],  # which layers to fine-tune
        bias           = "none",
        use_gradient_checkpointing = "unsloth",  # saves VRAM
        random_state   = 42,
    )

    print("Model loaded successfully.")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


# ============================================================
# STEP 4 — TRAINING
# ============================================================

def train(model, tokenizer, dataset):
    print("\nStarting fine-tuning...")

    trainer = SFTTrainer(
        model        = model,
        tokenizer    = tokenizer,
        train_dataset= dataset,
        dataset_text_field = "text",
        max_seq_length     = CONFIG["max_seq_length"],
        packing            = False,  # set True to pack short examples together

        args = TrainingArguments(
            per_device_train_batch_size    = CONFIG["batch_size"],
            gradient_accumulation_steps    = CONFIG["gradient_accumulation"],
            num_train_epochs               = CONFIG["epochs"],
            learning_rate                  = CONFIG["learning_rate"],
            warmup_steps                   = CONFIG["warmup_steps"],
            lr_scheduler_type              = CONFIG["lr_scheduler"],
            optim                          = CONFIG["optimizer"],
            weight_decay                   = CONFIG["weight_decay"],
            neftune_noise_alpha            = 5, # Adds noise to embeddings to improve instruction following

            # Precision — bf16 is faster on RTX 30/40/50 series
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),

            logging_steps    = 10,    # print loss every 10 steps
            save_steps       = 100,   # save checkpoint every 100 steps
            output_dir       = CONFIG["output_dir"],
            seed             = 42,
            report_to        = "none",  # disable wandb logging
        ),
    )

    # Show GPU stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_stats.name}")
    print(f"Total VRAM: {gpu_stats.total_memory / 1e9:.2f} GB")

    # Train
    trainer_stats = trainer.train()

    print(f"\nTraining complete.")
    print(f"Time taken: {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes")
    print(f"Final loss: {trainer_stats.metrics['train_loss']:.4f}")

    return model


# ============================================================
# STEP 5 — SAVE AND EXPORT
# ============================================================

def save_and_export(model, tokenizer):

    # Save the LoRA adapter weights
    print(f"\nSaving LoRA adapters to {CONFIG['output_dir']}...")
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])

    # Merge LoRA into base model and save full model
    print(f"Saving merged model to {CONFIG['final_model_dir']}...")
    model.save_pretrained_merged(
        CONFIG["final_model_dir"],
        tokenizer,
        save_method = "merged_16bit",
    )

    # Export to GGUF for llama.cpp on mobile
    print(f"Exporting to GGUF ({CONFIG['gguf_quantization']}) for mobile...")
    model.save_pretrained_gguf(
        CONFIG["gguf_output_dir"],
        tokenizer,
        quantization_method = CONFIG["gguf_quantization"],  # q4_k_m
    )

    gguf_file = os.path.join(
        CONFIG["gguf_output_dir"],
        f"model-{CONFIG['gguf_quantization'].upper()}.gguf"
    )
    print(f"\nGGUF model ready: {gguf_file}")
    print("Copy this file to your React Native app: assets/models/")


# ============================================================
# STEP 6 — QUICK TEST BEFORE EXPORT
# ============================================================

def test_model(model, tokenizer):
    print("\nRunning quick test...")

    FastLanguageModel.for_inference(model)  # switch to inference mode

    test_questions = [
        "Someone stopped breathing",
        "There is an earthquake happening",
        "Severe bleeding from arm won't stop",
    ]

    for question in test_questions:
        prompt = format_prompt(question, "")  # empty output = model completes it
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 200,
                temperature    = 0.1,   # low temperature = more deterministic
                repetition_penalty = 1.15, # prevents looping text in stressful edge cases
                top_p          = 0.9,   # nucleus sampling
                do_sample      = True,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the model's response
        response = response.split("<start_of_turn>model")[-1].strip()

        print(f"\nQ: {question}")
        print(f"A: {response}")
        print("-" * 50)


# ============================================================
# MAIN — RUN EVERYTHING
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CrisisNet — Gemma 4 E2B Fine-Tuning")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected. This script requires a CUDA GPU.")
        print("Make sure your RTX 5060 drivers and CUDA toolkit are installed.")
        exit(1)

    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    # Run pipeline
    dataset          = build_combined_dataset()
    model, tokenizer = load_model()
    model            = train(model, tokenizer, dataset)
    test_model(model, tokenizer)
    save_and_export(model, tokenizer)

    print("\n" + "=" * 60)
    print("Fine-tuning complete.")
    print(f"Your GGUF model is in: {CONFIG['gguf_output_dir']}")
    print("Drop it into: assets/models/ in your React Native project")
    print("=" * 60)