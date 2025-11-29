#  AUTOMATED DATA PREPROCESSING USING LLM INTEGRATED MULTI AGENT FARMEWORK
##  Project Description 

#### This project implements an Agentic AI–powered data cleaning system that automatically analyzes, plans, and cleans raw datasets using a coordinated set of intelligent agents. Instead of static rule-based preprocessing, this system uses a Controller Agent that dynamically selects appropriate cleaning steps based on dataset characteristics—making the entire process adaptive, explainable, and efficient.

#### The system is integrated into a Streamlit UI, enabling users to upload messy CSV files and receive a structured, cleaned output along with a detailed log of applied transformations.
### This project is built for:
- ML engineers
- Data scientists
- Researchers
- Students
#### Or Anyone who needs clean, standardized data quickly
#### The system completely offline after installing Ollama, or else it requires an API integration like openAI through API keys to run online.

##  Key Features
###  Agentic AI Pipeline
#### A Controller Agent plans the cleaning steps based on dataset summary.
### Specialized Cleaning Agents
- Missing Value Agent
- Outlier Correction Agent
- Type Correction Agent
- Duplicate Removal Agent
- Semantic Normalizer Agent (LLM-powered)
###  Local LLM Integration (Ollama)
#### Used ollama's 8B (8 billion) lightweight model for efficient and seamless integration through which the optimal preprocessing sequence could be generated.
###  Full Explainability Log
#### Every preprocessing step is recorded with:
- step name
- reason
- shape before
- shape after
###  Diagramatic Visualization
#### Shows the pipeline flow clearly.
###  Streamlit UI
#### Upload, preprocess, view logs, download output CSV.
