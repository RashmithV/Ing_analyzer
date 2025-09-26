import os
import json
import streamlit as st
import pandas as pd
import re
from text_extract import extract_text_from_image
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CEREBRAS_API_KEY")
# --- Config ---
TEMP_IMG = "temp_upload.jpg"
DB_FILE = "ingredient_db.json"
CEREBRAS_MODEL = "gpt-oss-120b"

# --- Initialize Cerebras client ---
client = Cerebras(api_key=api_key)

# --- Streamlit page config ---
st.set_page_config(page_title="Ingredient Health Analyzer", layout="wide")
st.title("🧴 Ingredient Health Analyzer")
st.write("Upload a product image → Extract text via OCR → Analyze ingredients for health impact.")

# --- Helpers ---
def clean_think_tags(text: str) -> str:
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text.replace(text[start:end], "").strip()
    return text

def clean_json_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text

# --- Database helpers ---
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def get_from_db(db, ingredient):
    return db.get(ingredient.lower())

def update_db(db, ingredient, classification, reason):
    db[ingredient.lower()] = {"Classification": classification, "Reason": reason}
    save_db(db)

# --- Ingredient splitter ---
def split_ingredients(text: str):
    ingredients = re.split(r",|;|/", text)
    return [ing.strip() for ing in ingredients if ing.strip() and len(ing) > 2 and not ing.lower().startswith("made in")]

# --- Cerebras query ---
def query_cerebras(prompt: str) -> str:
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        model=CEREBRAS_MODEL,
        stream=True,
        max_completion_tokens=20000,
        temperature=0.7,
        top_p=0.8
    )
    output = ""
    for chunk in stream:
        output += chunk.choices[0].delta.content or ""
    return output

# --- File uploader ---
uploaded_file = st.file_uploader("📷 Upload product image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open(TEMP_IMG, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(TEMP_IMG, caption="Uploaded Image", use_column_width=True)

    if st.button("🔎 Extract & Analyze Ingredients"):
        # --- OCR ---
        with st.spinner("Extracting ingredients from the image..."):
            ocr_text = extract_text_from_image(TEMP_IMG)
        st.subheader("📄 Extracted Text")
        st.text_area("OCR Output", ocr_text, height=200)

        # --- Load DB ---
        db = load_db()
        final_items = []

        # --- Split OCR text into candidate ingredients ---
        candidate_ingredients = split_ingredients(ocr_text)

        # --- Separate known and unknown ingredients ---
        unknown_ingredients = []
        for ing in candidate_ingredients:
            db_entry = get_from_db(db, ing)
            if db_entry:
                final_items.append({
                    "Ingredient": ing,
                    "Classification": db_entry["Classification"],
                    "Reason": db_entry["Reason"]
                })
            else:
                unknown_ingredients.append(ing)

        # --- Query Cerebras only for unknown ingredients ---
        if unknown_ingredients:
            prompt_text = (
                "Analyze the following ingredients. Classify each as 'Safe', "
                "'Caution advised', or 'Unsafe' for humans, and give a short reason. "
                "Respond ONLY in JSON array format with keys: "
                '"Ingredient", "Classification", "Reason".\n\n'
                "Ingredients:\n"
                + "\n".join(unknown_ingredients)
            )

            with st.spinner(f"Analyzing {len(unknown_ingredients)} new ingredient(s) with Cerebras..."):
                response_text = query_cerebras(prompt_text)
                response_text = clean_think_tags(response_text)
                response_text = clean_json_code_block(response_text)

            # --- Parse JSON safely ---
            try:
                data = json.loads(response_text)
            except Exception:
                st.error("❌ Could not parse JSON output. Showing raw text.")
                st.text(response_text)
                data = []

            # --- Add results to final items and DB ---
            for item in data:
                ing_name = item["Ingredient"].strip()
                final_items.append(item)
                update_db(db, ing_name, item["Classification"], item["Reason"])

        # --- Display DataFrame ---
        if final_items:
            df_final = pd.DataFrame(final_items)

            def highlight_classification(val):
                if isinstance(val, str):
                    v = val.lower()
                    if v == "safe":
                        return "background-color: #d4edda; color: #155724;"
                    elif v == "caution advised":
                        return "background-color: #fff3cd; color: #856404;"
                    elif v == "unsafe":
                        return "background-color: #f8d7da; color: #721c24;"
                return ""

            st.subheader("🔬 Ingredient Analysis")
            st.dataframe(df_final.style.applymap(highlight_classification, subset=["Classification"]), use_container_width=True)

            # --- Downloads ---
            st.download_button(
                "💾 Download CSV",
                df_final.to_csv(index=False).encode("utf-8"),
                "ingredients_analysis.csv",
                "text/csv"
            )
            st.download_button(
                "💾 Download JSON",
                df_final.to_json(orient="records", indent=2).encode("utf-8"),
                "ingredients_analysis.json",
                "application/json"
            )

st.markdown("---")
st.caption("Built with EasyOCR + Cerebras + Streamlit. Ingredients are cached locally in ingredient_db.json to reduce repeated AI calls.")