Hi there, I'm Ose


import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Occupation Classification QA Tool", layout="wide")

st.title("📊 Occupation Classification QA Tool")
st.markdown("""
Upload your labour force survey data to check if the enumerator-assigned ISCO codes match the predicted ones.
This tool helps speed up quality assurance and reduce manual checking.
""")

# File upload
uploaded_file = st.file_uploader("Upload survey data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Column selection
    st.subheader("🧭 Map your columns")
    text_col = st.selectbox("Select occupation text column", df.columns)
    code_col = st.selectbox("Select assigned ISCO code column", df.columns)

    # Run classification check
    if st.button("🔍 Run Classification Check"):
        with st.spinner("Classifying and checking..."):
            # Mock classification and comparison
            def mock_predict(text):
                # Simulate a predicted ISCO code and confidence
                return random.choice(["2112", "3114", "4221", "5211"]), round(random.uniform(0.7, 0.99), 2)

            df["predicted_code"] = df[text_col].apply(lambda x: mock_predict(x)[0])
            df["confidence"] = df[text_col].apply(lambda x: mock_predict(x)[1])
            df["match"] = df.apply(lambda row: row[code_col] == row["predicted_code"], axis=1)

            # Summary
            st.success("Classification complete.")
            st.subheader("📈 Summary")
            total = len(df)
            correct = df["match"].sum()
            flagged = total - correct
            st.metric("Records Checked", total)
            st.metric("Matches", correct)
            st.metric("Flagged for Review", flagged)

            # Show results
            st.subheader("🔎 Flagged Records")
            st.dataframe(df[df["match"] == False][[text_col, code_col, "predicted_code", "confidence"]])

            # Download
            st.subheader("⬇️ Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download full results as CSV",
                data=csv,
                file_name="classified_results.csv",
                mime="text/csv"
            )

# Optional help section
with st.expander("ℹ️ Help"):
    st.markdown("""
    - Make sure your file includes both the free-text occupation field and the ISCO code.
    - This tool uses a simulated classification for demo purposes.  
    - Actual deployment will include OpenAI-based or ML-trained classification.
    """)

