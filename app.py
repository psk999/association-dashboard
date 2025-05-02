import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from io import StringIO

st.set_page_config(page_title="Association Rule Mining Dashboard", layout="wide")
st.title("🧠 Association Rule Mining Dashboard")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    cols = df.columns.tolist()
    antecedents = st.multiselect("Select Antecedent Variables", cols)
    consequents = st.multiselect("Select Consequent Variables", cols)

    if antecedents and consequents:
        encoded_df = pd.get_dummies(df[antecedents + consequents])
        frequent_itemsets = apriori(encoded_df, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        lift_range = st.slider("Lift Range", 0.5, 5.0, (1.0, 2.0))
        filtered = rules[(rules['lift'] >= lift_range[0]) & (rules['lift'] <= lift_range[1])]

        st.write(f"### {len(filtered)} Filtered Rules")
        st.dataframe(filtered)
