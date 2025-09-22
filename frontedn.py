import os
import json
import streamlit as st
import pandas as pd
from langchain_ollama import OllamaLLM
from text_extract import extract_text_from_image  # <-- use the new function
import csv
from io import StringIO

# --- Config ---
OUTPUT_FILE = "ingredients_output.json"
TEMP_IMG = "temp_upload.jpg"

# Initialize Ollama LLM
model = OllamaLLM(model="gemma3:1b" )

st.set_page_config(page_title="Ingredient Health Analyzer", layout="wide")

st.title("🧴 Ingredient Health Analyzer")
st.write("Upload a product image → Extract text via OCR → Analyze ingredients for health impact.")

# --- File uploader for images ---
uploaded_file = st.file_uploader("📷 Upload product image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file temporarily
    with open(TEMP_IMG, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(TEMP_IMG, caption="Uploaded Image", use_container_width=True)

    if st.button("🔎 Extract & Analyze Ingredients"):
        with st.spinner("Extracting ingredients from the image..."):
            ocr_text = extract_text_from_image(TEMP_IMG)

        st.subheader("📄 Extracted Text from the image")
        st.text_area("Output", ocr_text, height=200)

        with st.spinner("Analyzing with LLM..."):
            prompt = f"""
            Here is OCR text of ingredients. Analyze and check if the ingredients are healthy or unhealthy for humans.
            Give output in table format with columns: Ingredient, Healthy/Unhealthy, Reason.
            remove text which is not an ingredient. 

            OCR Text:
            {ocr_text}
            """
            response_text = model.invoke(prompt)
        if "<think>" in response_text and "</think>" in response_text:
            start = response_text.find("<think>")
            end = response_text.find("</think>") + len("</think>")
            response_text = response_text.replace(response_text[start:end], "").strip()            

        # Try parsing JSON
        df = None
        try:
            # Try direct JSON parsing
            data = json.loads(response_text)
            df = pd.DataFrame(data)
        except Exception:
            st.warning("⚠️ Could not parse structured table. Trying to reformat with LLM...")

            # --- Ask LLM again just to reformat into JSON ---
            reformat_prompt = f"""
            Take the following text and reformat it into a correct table.

            Text to reformat:
            {response_text}
            """

            clean_output = model.invoke(reformat_prompt)

            try:
                data = json.loads(clean_output)
                df = pd.DataFrame(data)
            except Exception:
                st.error("❌ Still could not parse response. Showing raw output.")
                st.text(response_text)

        if df is not None:
            st.success("✅ Analysis complete!")

            # Apply color highlighting
            def highlight_health(val):
                if isinstance(val, str):
                    if val.lower() == "healthy":
                        return "background-color: #d4edda; color: #155724;"
                    elif val.lower() == "unhealthy":
                        return "background-color: #f8d7da; color: #721c24;"
                return ""

            styled_df = df.style.applymap(highlight_health, subset=["Healthy/Unhealthy"])
            st.dataframe(styled_df, use_container_width=True)