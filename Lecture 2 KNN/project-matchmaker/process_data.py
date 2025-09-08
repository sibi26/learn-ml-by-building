#!/usr/bin/env python3
"""
Process Google Forms CSV data into JSON for Project Matchmaker visualization.
Usage: python process_data.py responses.csv

Requirements:
- pip install pandas openai python-dotenv
- Create .env file with OPENAI_API_KEY=your_key_here
"""

import pandas as pd
import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_responses(csv_file):
    """Process student responses from CSV to JSON with dimensions and categories."""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Read CSV
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Prepare all responses for batch processing
    all_responses = []
    for _, row in df.iterrows():
        # Collect survey responses (adjust column names as needed)
        responses = {
            'email': row.get('VT Email ID', ''),  # Just the ID part
            'name': row.get('Full Name', ''),
            'application_area': row.get('Which ML/AI application area interests you most?', ''),
            'team_needed': row.get('How many people are you looking for to form a team?', ''),
            'project_approach': row.get('Describe your ideal ML project approach', ''),
            'experience': row.get('Describe your ML/AI experience and expertise', ''),
            'problem_type': row.get('What type of ML problem excites you more?', ''),
            'project_ideas': row.get('Project ideas or topics of interest', ''),
            'collaboration_style': row.get('How do you prefer to collaborate?', ''),
            'contact': row.get('Preferred contact method for team formation', '')
        }
        all_responses.append(responses)
    
    print(f"Processing {len(all_responses)} student responses...")
    
    # Create prompt for OpenAI
    prompt = f"""
    Analyze these student survey responses and convert each student into:
    
    1. Multiple numerical dimensions (range -1 to 1):
       - theory_implementation: -1 (pure theory/math focus) to 1 (pure implementation/engineering)
       - research_industry: -1 (academic research focus) to 1 (industry/product focus)
       - structured_exploratory: -1 (prefers structured problems) to 1 (prefers exploratory work)
       - solo_collaborative: -1 (prefers working alone) to 1 (prefers team collaboration)
       - depth_breadth: -1 (deep focus on one area) to 1 (broad interests across areas)
       - plan_iterate: -1 (careful planning first) to 1 (quick iteration and testing)
       - data_model: -1 (data/analysis focused) to 1 (model/algorithm focused)
       - foundational_applied: -1 (building foundations) to 1 (immediate applications)
       
    Base these dimensions on their project approach, experience description, collaboration preferences, and project ideas.
    
    2. Categorical features:
       - team_status: Based on team_needed field:
         * "solo" if they chose "None - I'm working solo OR I already have a team"
         * "seeking_small" if looking for 1 person
         * "seeking_medium" if looking for 2 people  
         * "seeking_large" if looking for 3 people
       - main_interest: Extract from application_area, use exactly one of: "nlp", "computer_vision", "healthcare", "robotics", "energy", "cybersecurity", "general"
       - experience_level: Infer from experience field: "beginner", "intermediate", "advanced"
       - problem_preference: From problem_type field: "structured", "exploratory", or "both"
       - collaboration_style: From collaboration field: "meetings", "divided", or "flexible"
       - has_project_idea: true if project_ideas contains specific ideas, false if vague/empty
    
    Return a JSON array with one object per student containing:
    - email (add @vt.edu to make full email)
    - name, contact (preserve from input)
    - theory_implementation, research_industry, structured_exploratory, solo_collaborative, 
      depth_breadth, plan_iterate, data_model, foundational_applied (all numerical dimension values)
    - team_status, main_interest, experience_level, problem_preference, collaboration_style, has_project_idea
    - original_responses (preserve all responses as an object)
    
    Be consistent in your dimensional mappings across all students. Someone with heavy research/paper focus should be negative on research_industry axis, while someone focused on building products should be positive.
    
    Student responses:
    {json.dumps(all_responses, indent=2)}
    """
    
    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that processes student survey data into structured formats."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for consistency
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        processed_students = json.loads(response.choices[0].message.content)
        
        # Structure the final output
        output = {
            "dimensions": [
                "theory_implementation",
                "research_industry", 
                "structured_exploratory",
                "solo_collaborative",
                "depth_breadth",
                "plan_iterate",
                "data_model",
                "foundational_applied"
            ],
            "dimension_descriptions": {
                "theory_implementation": "Theory/Math ← → Implementation/Engineering",
                "research_industry": "Research/Academic ← → Industry/Product",
                "structured_exploratory": "Structured Problems ← → Exploratory/Open-ended",
                "solo_collaborative": "Solo Work ← → Team Collaboration", 
                "depth_breadth": "Deep Focus ← → Broad Interests",
                "plan_iterate": "Plan First ← → Iterate Quickly",
                "data_model": "Data/Analysis First ← → Model/Algorithm First",
                "foundational_applied": "Build Foundations ← → Applied Solutions"
            },
            "categorical_features": {
                "team_status": {
                    "name": "Team Formation Status",
                    "values": ["solo", "seeking_small", "seeking_medium", "seeking_large"],
                    "colors": ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b"]  # Purple, Blue, Green, Orange
                },
                "main_interest": {
                    "name": "Primary Interest Area",
                    "values": ["nlp", "computer_vision", "healthcare", "robotics", "energy", "cybersecurity", "general"],
                    "colors": ["#ef4444", "#3b82f6", "#10b981", "#a855f7", "#f59e0b", "#6366f1", "#6b7280"]
                },
                "experience_level": {
                    "name": "Experience Level",
                    "values": ["beginner", "intermediate", "advanced"],
                    "colors": ["#34d399", "#3b82f6", "#a855f7"]  # Light green, Blue, Purple
                },
                "problem_preference": {
                    "name": "Problem Type Preference",
                    "values": ["structured", "exploratory", "both"],
                    "colors": ["#3b82f6", "#f59e0b", "#10b981"]  # Blue, Orange, Green
                },
                "collaboration_style": {
                    "name": "Collaboration Style",
                    "values": ["meetings", "divided", "flexible"],
                    "colors": ["#ef4444", "#3b82f6", "#10b981"]  # Red, Blue, Green
                },
                "has_project_idea": {
                    "name": "Has Specific Project Idea",
                    "values": [true, false],
                    "colors": ["#10b981", "#6b7280"]  # Green, Gray
                }
            },
            "students": processed_students
        }
        
        # Save to JSON file
        output_file = 'student_data.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Successfully processed {len(processed_students)} students")
        print(f"✓ Output saved to {output_file}")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"- Dimensions: {', '.join(output['dimensions'].values())}")
        print(f"- Categorical features: {', '.join(output['categorical_features'].keys())}")
        
    except Exception as e:
        print(f"Error processing responses: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_data.py <csv_file>")
        print("Example: python process_data.py responses.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Create a .env file with: OPENAI_API_KEY=your_key_here")
        sys.exit(1)
    
    process_responses(csv_file)

if __name__ == "__main__":
    main()