import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from datetime import datetime

# Uncomment if not using pipenv
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

# Configuration and Settings
DB_FAISS_PATH = "vectorstore/db_faiss"

# Set page configuration
st.set_page_config(
    page_title="NeuroMate - AI Neurosurgical Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-themed styling
st.markdown("""
    <style>
    .css-1d391kg {
        padding-top: 0rem;
    }
    .stChat {
        padding: 20px;
        border-radius: 15px;
        background-color: #f0f2f6;
    }
    .source-doc {
        padding: 10px;
        border-radius: 5px;
        background-color: #ffffff;
        margin: 5px 0;
        border-left: 3px solid #0066cc;
    }
    .chat-timestamp {
        font-size: 0.8em;
        color: #666;
    }
    .patient-info {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #0066cc;
    }
    .medical-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, 
                          input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm

def format_source_documents(source_documents):
    formatted_sources = []
    for i, doc in enumerate(source_documents, 1):
        source_info = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'source_number': i
        }
        formatted_sources.append(source_info)
    return formatted_sources

def display_source_document(source):
    with st.container():
        st.markdown(f"""
        <div class="source-doc">
            <strong>Source {source['source_number']}</strong><br>
            Content: {source['content']}<br>
            {' | '.join([f'{k}: {v}' for k, v in source['metadata'].items()])}
        </div>
        """, unsafe_allow_html=True)

def init_patient_info():
    default_fields = [
        'patient_id', 'patient_age', 'patient_gender', 'patient_diagnosis',
        'patient_history', 'patient_medications', 'patient_allergies',
        'patient_imaging', 'patient_labs'
    ]
    for field in default_fields:
        if field not in st.session_state:
            st.session_state[field] = ""

def main():
    # Initialize patient info
    init_patient_info()

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150", caption="NeuroMate")
        st.title("🧠 NeuroMate Settings")
        
        # Chat Settings
        st.markdown("### Chat Settings")
        temperature = st.slider("AI Temperature", 0.0, 1.0, 0.5, 0.1)
        show_sources = st.checkbox("Show Reference Sources", True)
        include_patient_info = st.checkbox("Enable Patient Context", False)
        
        # Optional Patient Information
        if include_patient_info:
            st.markdown("### Patient Information (Optional)")
            with st.expander("Add Patient Details", expanded=False):
                with st.form("patient_info_form"):
                    st.session_state.patient_id = st.text_input("Patient ID", st.session_state.patient_id)
                    st.session_state.patient_age = st.number_input("Age", 0, 150, value=int(st.session_state.patient_age) if st.session_state.patient_age else 0)
                    st.session_state.patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
                    st.session_state.patient_diagnosis = st.text_area("Primary Diagnosis", st.session_state.patient_diagnosis)
                    st.session_state.patient_history = st.text_area("Relevant Medical History", st.session_state.patient_history)
                    st.session_state.patient_medications = st.text_area("Current Medications", st.session_state.patient_medications)
                    st.session_state.patient_allergies = st.text_area("Allergies", st.session_state.patient_allergies)
                    st.session_state.patient_imaging = st.text_area("Recent Imaging Findings", st.session_state.patient_imaging)
                    st.session_state.patient_labs = st.text_area("Relevant Lab Results", st.session_state.patient_labs)
                    submit_button = st.form_submit_button("Update Patient Info")
        
        st.markdown("---")
        st.markdown("### About NeuroMate")
        st.markdown("""
        NeuroMate is your AI-powered neurosurgical assistant, designed to help with:
        - Clinical decision support
        - Treatment planning
        - Protocol references
        - Literature consultation
        """)
        
        st.markdown("---")
        st.markdown("""
        <div class='disclaimer'>
        ⚠️ This AI assistant is for informational purposes only and should not replace 
        professional medical judgment. Always verify information and consult with 
        appropriate medical professionals.
        </div>
        """, unsafe_allow_html=True)

    # Main content area
    st.title("🧠 NeuroMate - AI Neurosurgical Assistant")
    
    # Display current patient context if available and enabled
    if include_patient_info and st.session_state.patient_id:
        with st.expander("Current Patient Context", expanded=False):
            st.markdown("""<div class='patient-info'>""", unsafe_allow_html=True)
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**Patient ID:** {st.session_state.patient_id}")
                st.markdown(f"**Age:** {st.session_state.patient_age}")
                st.markdown(f"**Gender:** {st.session_state.patient_gender}")
                st.markdown(f"**Primary Diagnosis:** {st.session_state.patient_diagnosis}")
                st.markdown(f"**Allergies:** {st.session_state.patient_allergies}")
            with cols[1]:
                st.markdown(f"**Medical History:** {st.session_state.patient_history}")
                st.markdown(f"**Current Medications:** {st.session_state.patient_medications}")
                st.markdown(f"**Imaging Findings:** {st.session_state.patient_imaging}")
                st.markdown(f"**Lab Results:** {st.session_state.patient_labs}")
            st.markdown("</div>", unsafe_allow_html=True)

    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'timestamp' in message:
                st.markdown(f"<div class='chat-timestamp'>{message['timestamp']}</div>", 
                          unsafe_allow_html=True)
            if message.get('sources') and show_sources:
                st.markdown("---\n**Reference Sources:**")
                for source in message['sources']:
                    display_source_document(source)

    # Chat input
    prompt = st.chat_input("Ask your neurosurgical question...")
    
    if prompt:
        timestamp = datetime.now().strftime("%I:%M %p")
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
            'timestamp': timestamp
        })

        # Custom prompt template that includes patient context only if enabled
        if include_patient_info and st.session_state.patient_id:
            CUSTOM_PROMPT_TEMPLATE = """
            You are NeuroMate, an AI neurosurgical assistant. Use the provided context and your knowledge to answer the question.
            Consider the following patient information in your response:
            Patient Context:
            - Age: {st.session_state.patient_age}
            - Gender: {st.session_state.patient_gender}
            - Diagnosis: {st.session_state.patient_diagnosis}
            - Medical History: {st.session_state.patient_history}
            - Current Medications: {st.session_state.patient_medications}
            - Allergies: {st.session_state.patient_allergies}
            - Recent Imaging: {st.session_state.patient_imaging}
            - Lab Results: {st.session_state.patient_labs}

            Context: {context}
            Question: {question}

            Provide a clear, professional response. Include relevant medical considerations and any important caveats.
            """
        else:
            CUSTOM_PROMPT_TEMPLATE = """
            You are NeuroMate, an AI neurosurgical assistant. Use the provided context and your knowledge to answer the question.

            Context: {context}
            Question: {question}

            Provide a clear, professional response with relevant medical considerations.
            """

        HUGGINGFACE_REPO_ID = "google/flan-t5-small"
        #HF_TOKEN = os.environ.get("HF_TOKEN")
        HF_TOKEN = st.secrets["hf_token"]

        try:
            with st.spinner("Analyzing medical information..."):
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the medical knowledge base")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                
                formatted_sources = format_source_documents(source_documents)
                
                with st.chat_message('assistant'):
                    st.markdown(result)
                    if show_sources:
                        st.markdown("---\n**Reference Sources:**")
                        for source in formatted_sources:
                            display_source_document(source)

                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result,
                    'timestamp': datetime.now().strftime("%I:%M %p"),
                    'sources': formatted_sources
                })

        except Exception as e:
            st.error(f"Error in medical analysis: {str(e)}")

if __name__ == "__main__":
    main()