"""
Cat Classifier module for CatShop
Integrates Gemma-3 model for cat perspective classification and chat
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import random

# Cat categories and their properties
CAT_CATEGORIES = {
    "NAP_SURFACE": {"emoji": "😴", "color": "#9b59b6", "description": "Perfect for napping"},
    "HUNT_PLAY": {"emoji": "🎯", "color": "#e74c3c", "description": "Fun to hunt and play with"},
    "TERRITORY": {"emoji": "🏰", "color": "#3498db", "description": "Must mark as mine"},
    "GROOMING": {"emoji": "✨", "color": "#f39c12", "description": "Keeps me beautiful"},
    "CONSUMPTION": {"emoji": "🍽️", "color": "#27ae60", "description": "Food or water"},
    "DANGER": {"emoji": "⚠️", "color": "#c0392b", "description": "Scary! Must avoid!"},
    "IRRELEVANT": {"emoji": "😑", "color": "#95a5a6", "description": "Boring human stuff"}
}

class CatClassifier:
    def __init__(self, model_path="models/gemma-cat-lora"):
        """Initialize the cat classifier with Gemma-3 model"""
        self.model_path = Path(__file__).parent.parent / model_path  # Absolute path from project root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned Gemma-3 model"""
        try:
            # Load base model and tokenizer
            base_model = "google/gemma-3-270m"  # Use Gemma-3 270m
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Check if LoRA weights exist
            if self.model_path.exists():
                # Load with LoRA
                base_model_obj = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.model = PeftModel.from_pretrained(base_model_obj, str(self.model_path))
            else:
                # Fallback to base model
                print(f"Warning: LoRA weights not found at {self.model_path}, using base model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            print(f"✅ Cat model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Could not load Gemma model: {e}")
            print("Falling back to rule-based classification")
            self.model = None
    
    def classify_product(self, product_name, product_description=""):
        """Classify a product from cat perspective"""
        if self.model is None:
            return self.rule_based_classify(product_name, product_description)
        
        try:
            # Create classification prompt
            prompt = f"""You are a cat evaluating products. Classify this product into one of these categories:
- NAP_SURFACE: Things to sleep on (warm, soft, elevated)
- HUNT_PLAY: Things to chase or play with
- TERRITORY: Things to mark or claim
- GROOMING: Self-care items
- CONSUMPTION: Food and water
- DANGER: Scary things to avoid
- IRRELEVANT: Boring human stuff

Product: {product_name[:100]}
Category:"""
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
            
            # Calculate probabilities for each category
            category_tokens = {}
            for cat in CAT_CATEGORIES.keys():
                cat_tokens = self.tokenizer.encode(" " + cat, add_special_tokens=False)
                if cat_tokens:
                    category_tokens[cat] = cat_tokens[0]
            
            if category_tokens:
                cat_logits = torch.stack([logits[token_id] for token_id in category_tokens.values()])
                cat_probs = F.softmax(cat_logits / 0.7, dim=0)  # temperature=0.7
                
                probs = {}
                for cat, prob in zip(category_tokens.keys(), cat_probs):
                    probs[cat] = prob.item()
            else:
                # Fallback to uniform
                probs = {cat: 1.0/len(CAT_CATEGORIES) for cat in CAT_CATEGORIES}
            
            # Get top prediction
            top_cat = max(probs, key=probs.get)
            
            return {
                "category": top_cat,
                "confidence": probs[top_cat],
                "all_probabilities": probs,
                "emoji": CAT_CATEGORIES[top_cat]["emoji"],
                "color": CAT_CATEGORIES[top_cat]["color"],
                "description": CAT_CATEGORIES[top_cat]["description"]
            }
            
        except Exception as e:
            print(f"Model inference error: {e}")
            return self.rule_based_classify(product_name, product_description)
    
    def rule_based_classify(self, product_name, product_description=""):
        """Fallback rule-based classification"""
        text = f"{product_name} {product_description}".lower()
        
        # Simple keyword matching
        if any(w in text for w in ['laptop', 'computer', 'keyboard', 'monitor', 'bed', 'cushion', 'pillow']):
            category = 'NAP_SURFACE'
        elif any(w in text for w in ['toy', 'ball', 'feather', 'play', 'mouse', 'laser']):
            category = 'HUNT_PLAY'
        elif any(w in text for w in ['furniture', 'chair', 'sofa', 'scratch', 'post']):
            category = 'TERRITORY'
        elif any(w in text for w in ['vacuum', 'blender', 'loud', 'cleaner', 'spray']):
            category = 'DANGER'
        elif any(w in text for w in ['food', 'treat', 'bowl', 'water', 'kibble']):
            category = 'CONSUMPTION'
        elif any(w in text for w in ['brush', 'groom', 'shampoo', 'comb']):
            category = 'GROOMING'
        else:
            category = 'IRRELEVANT'
        
        # Add some randomness to confidence for realism
        confidence = 0.7 + random.random() * 0.25
        
        # Create fake probability distribution
        probs = {cat: 0.05 for cat in CAT_CATEGORIES}
        probs[category] = confidence
        remaining = 1.0 - confidence - 0.05 * (len(CAT_CATEGORIES) - 1)
        for cat in probs:
            if cat != category:
                probs[cat] += remaining / (len(CAT_CATEGORIES) - 1)
        
        return {
            "category": category,
            "confidence": confidence,
            "all_probabilities": probs,
            "emoji": CAT_CATEGORIES[category]["emoji"],
            "color": CAT_CATEGORIES[category]["color"],
            "description": CAT_CATEGORIES[category]["description"]
        }
    
    def chat_about_product(self, product_name, user_question):
        """Have a cat conversation about a product"""
        if self.model is None:
            return self.rule_based_chat(product_name, user_question)
        
        try:
            prompt = f"""You are a cat evaluating products for online shopping. Be playful, mention cat behaviors like napping, hunting, scratching, and give opinions from a cat's perspective. Keep responses under 100 words.

Product: {product_name}
Human: {user_question}
Cat:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the cat's response
            if "Cat:" in response:
                response = response.split("Cat:")[-1].strip()
            else:
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Chat generation error: {e}")
            return self.rule_based_chat(product_name, user_question)
    
    def rule_based_chat(self, product_name, user_question):
        """Fallback rule-based chat responses"""
        classification = self.rule_based_classify(product_name)
        category = classification["category"]
        
        responses = {
            "NAP_SURFACE": [
                f"*yawns* Oh, {product_name} looks purr-fect for my afternoon naps! The surface seems warm and comfortable. I claim it as my new throne!",
                f"Meow! This {product_name} would make an excellent napping spot. Much better than that silly cat bed the humans bought me.",
                f"*stretches* I can already imagine curling up on this {product_name}. The humans won't mind if I shed a little fur on it, right?"
            ],
            "HUNT_PLAY": [
                f"*eyes dilate* Ooh! {product_name} triggers my hunting instincts! I must pounce immediately!",
                f"*wiggles butt* This {product_name} looks like so much fun to chase! My inner predator is activated!",
                f"Mrow! Finally, something exciting! I shall hunt this {product_name} at 3 AM when the humans are sleeping."
            ],
            "TERRITORY": [
                f"*sniffs* This {product_name} clearly needs my scent on it. It's mine now, humans!",
                f"Hmm, {product_name} would be perfect for scratching. Much better than that expensive scratching post!",
                f"I must mark this {product_name} as part of my territory. *rubs cheek against screen*"
            ],
            "DANGER": [
                f"HISS! {product_name} is scary! *runs away and hides under the bed*",
                f"*ears flatten* No no no! Keep that terrifying {product_name} away from me!",
                f"*puffs up* That {product_name} is my mortal enemy! I shall observe it from a safe distance... like another room."
            ],
            "CONSUMPTION": [
                f"*licks lips* Is this {product_name} food-related? I'm suddenly interested... unless it's medicine disguised as a treat!",
                f"Meow? Does {product_name} contain treats? I deserve treats for being magnificent!",
                f"*sniffs suspiciously* This {product_name} better be the good stuff, not that healthy food the vet recommended."
            ],
            "GROOMING": [
                f"*grooms paw* I suppose {product_name} is acceptable for maintaining my glorious coat.",
                f"Meh, {product_name} might help me look even more beautiful, if that's even possible.",
                f"*yawns* Do I really need {product_name}? I'm already purr-fect! But I'll allow it."
            ],
            "IRRELEVANT": [
                f"*yawns* {product_name}? How boring. Wake me when there's something actually interesting.",
                f"*turns away* Another silly human thing. This {product_name} has nothing to do with me.",
                f"Mrow... {product_name} is just human nonsense. Where are my treats?"
            ]
        }
        
        import random
        return random.choice(responses.get(category, responses["IRRELEVANT"]))

# Global instance
cat_classifier = None

def get_cat_classifier():
    """Get or create the global cat classifier instance"""
    global cat_classifier
    if cat_classifier is None:
        cat_classifier = CatClassifier()
    return cat_classifier