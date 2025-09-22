import os
import json
from langchain_ollama import OllamaLLM

input_folder = "output"   # folder containing OCR text files
output_file = "ingredients_output.json"

# Initialize Ollama LLM
model = OllamaLLM(model="deepseek-r1:1.5b")

all_outputs = {}

# Loop over all text files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            ocr_text = f.read()
        
        # Create prompt with OCR text
        prompt = f"""
        We have here an OCR text of ingredients.Analyze the text and check if the ingredients are healthy or unhealthy for humans.
        Give output in table format with columns: Ingredient, Healthy/Unhealthy, Reason. 

        OCR Text:
        {ocr_text}
        """
        
        response_text = model.invoke(prompt)
        
        
print(response_text)
