import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
import plotly.graph_objects as go
from datetime import datetime
from ollama import Client
import json

# Initialize Ollama Client

try:
    client = Client(host="http://localhost:11434")
    MODEL_NAME = "llama3:8b"    
except Exception as e:
    st.error("⚠️ Ollama server not running. Please start it using:  'ollama serve'")
    st.stop()

#  Explainability Log

class ExplainabilityLog:
    def __init__(self):
        self.logs = []

    def add(self, step, reason, before_shape, after_shape):
        self.logs.append({
            "Step": step,
            "Reason": reason,
            "Before": str(before_shape),
            "After": str(after_shape)
        })

    def to_dataframe(self):
        return pd.DataFrame(self.logs)


#  Cleaning Agents

class MissingValueAgent:
    def process(self, df, explainer):
        before = df.shape
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            for col in missing_cols:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            explainer.add("Missing Value Handling",
                          f"Filled missing values in {missing_cols} using median/mode.",
                          before, df.shape)
        return df


class OutlierAgent:
    def process(self, df, explainer):
        before = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = np.clip(df[col], lower, upper)
                explainer.add("Outlier Correction",
                              f"Clipped {outliers} outliers in '{col}' using IQR.",
                              before, df.shape)
        return df


class TypeCorrectionAgent:
    def process(self, df, explainer):
        before = df.shape
        converted = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                converted.append(col)
            except:
                continue
        if converted:
            explainer.add("Type Correction",
                          f"Converted {converted} to numeric where possible.",
                          before, df.shape)
        return df


class DuplicateRemovalAgent:
    def process(self, df, explainer):
        before = df.shape
        df = df.drop_duplicates()
        after = df.shape
        if before != after:
            explainer.add("Duplicate Removal",
                          f"Removed {before[0]-after[0]} duplicate rows.",
                          before, after)
        return df


# Semantic Normalizer (Ollama)

class SemanticNormalizerAgent:
    def __init__(self, client):
        self.client = client
        self.cache = {}  # prevent re-querying same words

    def normalize_text(self, text):
        text = str(text).strip()
        if text in self.cache:
            return self.cache[text]

        try:
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You standardize short categorical terms (e.g., ENG→English, USA→United States). Keep outputs short."},
                    {"role": "user", "content": f"Normalize: {text}"}
                ],
                stream=False  # Important: avoid streaming generator responses
            )

            if isinstance(response, dict) and "message" in response:
                norm = response["message"]["content"].strip()
            elif hasattr(response, "message"):
                norm = response.message.content.strip()
            else:
                norm = text

            self.cache[text] = norm
            return norm

        except Exception as e:
            print("⚠️ Ollama error:", e)
            return text

    def process(self, df, explainer):
        before = df.shape
        object_cols = df.select_dtypes(include=['object']).columns
        normalized_cols = []

        for col in object_cols:
            unique_values = df[col].dropna().unique().tolist()
            if 1 < len(unique_values) < 30:
                df[col] = df[col].apply(self.normalize_text)
                normalized_cols.append(col)

        if normalized_cols:
            explainer.add("Semantic Normalization",
                          f"Standardized {normalized_cols} using Llama3 locally.",
                          before, df.shape)
        return df


# Controller Agent (Main Coordinator)

class ControllerAgent:
    def __init__(self, client):
        self.client = client
        self.agents = {
            "MissingValueAgent": MissingValueAgent(),
            "TypeCorrectionAgent": TypeCorrectionAgent(),
            "OutlierAgent": OutlierAgent(),
            "DuplicateRemovalAgent": DuplicateRemovalAgent(),
            "SemanticNormalizerAgent": SemanticNormalizerAgent(client),
        }

    def summarize_dataset(self, df):
        return {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing_count": int(df.isna().sum().sum()),
            "duplicate_count": int(df.duplicated().sum()),
            "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
            "object_cols": df.select_dtypes(include=['object']).columns.tolist()
        }

    def llm_decision(self, summary):
        """Use Ollama to make intelligent cleaning decisions."""
        prompt = f"""You are a data preprocessing planner.
Dataset summary:
- Rows: {summary['rows']}
- Columns: {summary['cols']}
- Missing values: {summary['missing_count']}
- Duplicates: {summary['duplicate_count']}
- Numeric columns: {summary['numeric_cols']}
- Object columns: {summary['object_cols']}

Choose the best preprocessing steps in order. Respond ONLY with a JSON list.
Example: ["MissingValueAgent", "TypeCorrectionAgent", "DuplicateRemovalAgent"]

Available agents:
- MissingValueAgent (for missing values)
- TypeCorrectionAgent (for type conversion)
- OutlierAgent (for outliers in numeric columns)
- DuplicateRemovalAgent (for duplicates)
- SemanticNormalizerAgent (for text standardization)

RESPOND ONLY WITH THE JSON LIST, NOTHING ELSE."""

        try:
            response = self.client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            
            # Extract content
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                content = str(response)
            
            # Parse JSON from response
            content = content.strip()
            
            # Try to extract JSON if it's embedded in text
            if "[" in content and "]" in content:
                start = content.find("[")
                end = content.rfind("]") + 1
                json_str = content[start:end]
                plan = json.loads(json_str)
                
                # Validate that plan contains valid agent names
                valid_agents = [
                    "MissingValueAgent", "TypeCorrectionAgent", "OutlierAgent",
                    "DuplicateRemovalAgent", "SemanticNormalizerAgent"
                ]
                plan = [agent for agent in plan if agent in valid_agents]
                
                if plan:  # If we got valid agents, use the plan
                    st.write(f"✅ LLM Plan: {plan}")
                    return plan
            
        except Exception as e:
            st.warning(f"⚠️ LLM parsing error: {str(e)}")
        
        # Fallback: intelligent default based on data
        st.info("Using default cleaning sequence...")
        plan = []
        if summary['missing_count'] > 0:
            plan.append("MissingValueAgent")
        if summary['object_cols']:
            plan.append("TypeCorrectionAgent")
        if summary['numeric_cols']:
            plan.append("OutlierAgent")
        if summary['duplicate_count'] > 0:
            plan.append("DuplicateRemovalAgent")
        
        if not plan:
            plan = ["MissingValueAgent", "TypeCorrectionAgent", "DuplicateRemovalAgent"]
        
        return plan


    def analyze_and_clean(self, df):
        explainer = ExplainabilityLog()
        summary = self.summarize_dataset(df)
        plan = self.llm_decision(summary)
        explainer.add("Controller Decision", f"Chosen steps: {plan}", df.shape, df.shape)

        for step in plan:
            if step in self.agents:
                df = self.agents[step].process(df, explainer)

        return df, explainer

#  Visualization 
def plot_flow(log_df):
    if log_df.empty:
        return go.Figure()
    steps = ["Input"] + log_df["Step"].tolist() + ["Output"]
    sizes = [eval(log_df.iloc[i]["Before"])[0] for i in range(len(log_df))] + [eval(log_df.iloc[-1]["After"])[0]]
    sources = list(range(len(steps) - 1))
    targets = list(range(1, len(steps)))
    values = [abs(sizes[i] - sizes[i + 1]) + 1 for i in range(len(sizes) - 1)]

    fig = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=20, label=steps, color="lightblue"),
        link=dict(source=sources, target=targets, value=values)
    ))
    fig.update_layout(title_text="🧭 Data Cleaning Flow", font_size=12)
    return fig


#  Streamlit UI

st.set_page_config(page_title=" Agentic AI Based Data Cleaner", layout="wide")
st.title(" Agentic AI based Data Preprocessor ")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("📊 Preview (First 10 Rows)")
    st.dataframe(df.head(10))

    if st.button("🚀 Run Smart Cleaning"):
        with st.spinner("🤖 Controller Agent planning and executing pipeline..."):
            controller = ControllerAgent(client)
            cleaned_df, explainer = controller.analyze_and_clean(df.copy())

        st.success("✅ Cleaning complete!")
        st.dataframe(cleaned_df.head())

        st.subheader("🧠 Explainability Log")
        log_df = explainer.to_dataframe()
        st.dataframe(log_df)

        st.subheader("🔍 Cleaning Flow Visualization")
        fig = plot_flow(log_df)
        st.plotly_chart(fig, use_container_width=True)

        # Save cleaned file
        original_name = uploaded_file.name.split(".")[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_filename = f"{original_name}_cleaned_{timestamp}.csv"
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", cleaned_filename)
        cleaned_df.to_csv(downloads_path, index=False)

        buffer = BytesIO()
        cleaned_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("⬇️ Download Cleaned CSV", data=buffer, file_name=cleaned_filename, mime="text/csv")

        st.info(f"📁 File also saved locally: {downloads_path}")

else:
    st.info("👆 Upload a CSV to begin.")
