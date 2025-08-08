import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Insurance Claim Processor", layout="wide")
st.title("🤖 Insurance Claim Processor UI")
st.markdown("""
This app lets you test insurance claim scenarios against policy documents using semantic search and reasoning.
""")

with st.sidebar:
    st.header("API Controls")
    api_url = st.text_input("API URL", API_URL)
    st.info("Start the FastAPI server before using the UI.")

st.subheader("Submit a Claim Query")
query = st.text_area("Enter your claim scenario (natural language)",
    "46-year-old male, knee surgery in Pune, 3-month-old insurance policy")

if st.button("Process Claim"):
    with st.spinner("Processing claim..."):
        try:
            res = requests.post(f"{api_url}/process_claim", json={"query": query, "return_json": True})
            result = res.json()
            st.success(f"Decision: {result.get('decision','-').upper()}")
            st.markdown(f"**Justification:** {result.get('justification','-')}")
            st.markdown(f"**Amount:** {result.get('amount','-')}")
            st.markdown("**Mapped Clauses:**")
            for clause in result.get("mapped_clauses", []):
                with st.expander(f"{clause['clause_id']}"):
                    st.write(clause["text"])
                    st.caption(f"Relevance: {clause['relevance']}")
            st.markdown("**Full JSON Output:**")
            st.code(json.dumps(result, indent=2), language="json")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.subheader("Analyze Query Only (Entity Extraction)")
if st.button("Analyze Query"):
    with st.spinner("Extracting entities..."):
        try:
            res = requests.get(f"{api_url}/analyze_query", params={"query": query})
            result = res.json()
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.subheader("Semantic Clause Search")
top_k = st.slider("Top K Clauses", 1, 20, 5)
if st.button("Search Clauses"):
    with st.spinner("Searching for relevant clauses..."):
        try:
            res = requests.get(f"{api_url}/search_clauses", params={"query": query, "top_k": top_k})
            result = res.json()
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
if st.button("Show Policy Summary"):
    with st.spinner("Fetching policy summary..."):
        try:
            res = requests.get(f"{api_url}/policy_summary")
            result = res.json()
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")
