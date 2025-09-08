#!/usr/bin/env python3
"""Generate conversation and explanation training data using Gemma-3-270m"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm
from pathlib import Path

def generate_training_data():
    # Use correct model name
    print("Loading Gemma-3-270m...")
    model_name = "google/gemma-3-270m"  # FIXED!
    
    # Load model with proper device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Resolve paths relative to this script so it works from any CWD
    script_dir = Path(__file__).resolve().parent
    processed_dir = script_dir / 'processed'

    # Load products
    with open(processed_dir / 'cat_products.json', 'r') as f:
        products = json.load(f)
    
    conversation_examples = []
    explanation_examples = []
    
    print("Generating training examples...")
    for product in tqdm(products[:500]):  # Adjust number as needed
        name = product['name']
        category = product['cat_category']
        
        # Generate conversation examples - 3 variations per product
        conv_prompts = [
            f"Human: I bought a {name}. What will my cat think?\nAssistant:",
            f"Human: How would a cat react to {name}?\nAssistant:",
            f"Human: My cat saw my new {name}. What's their opinion?\nAssistant:"
        ]
        
        for prompt in conv_prompts[:1]:  # One per product to save time
            inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            conversation_examples.append({
                "prompt": prompt,
                "completion": f" Based on cat behavior, {response}",
                "category": category,
                "product_name": name
            })
        
        # Generate explanation with better prompt
        expl_prompt = f"Explain why a cat would categorize {name} as {category.lower().replace('_', ' ')}:"
        inputs = tokenizer(expl_prompt, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = explanation[len(expl_prompt):].strip()
        
        explanation_examples.append({
            "prompt": expl_prompt,
            "completion": f" {explanation}",
            "category": category,
            "product_name": name
        })
    
    # Save data
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(processed_dir / 'conversation_examples.json', 'w') as f:
        json.dump(conversation_examples, f, indent=2)
    
    with open(processed_dir / 'explanation_examples.json', 'w') as f:
        json.dump(explanation_examples, f, indent=2)
    
    print(f"✅ Generated {len(conversation_examples)} conversations")
    print(f"✅ Generated {len(explanation_examples)} explanations")

if __name__ == "__main__":
    generate_training_data()