
# RAG Application on Paul Graham Essay

## Overview
This application is built to perform in-depth analysis and insight generation on essays by Paul Graham using Retrieval-Augmented Generation (RAG). By integrating advanced retrieval mechanisms with generative AI, the system efficiently extracts relevant contextual information and synthesizes meaningful interpretations. This enhances the understanding of Graham’s writings by identifying key themes, extracting nuanced insights, and generating novel perspectives based on retrieved knowledge.

## Features
- **Intelligent Text Retrieva**: Implements a robust retrieval system using vector search and semantic indexing to efficiently locate and extract the most relevant sections from Paul Graham’s essays.
- **Context-Aware Text Generation**: Utilizes advanced language models to generate coherent, contextually relevant responses by leveraging retrieved text.
- **Interactive Query System**: Allows users to input questions or topics of interest, dynamically retrieving and generating responses in real time.

## Streamlit Link
Get started by exploring the Streamlit application [here](https://graham-bot-419.streamlit.app/).

## Installation
To install the required dependencies, follow these steps: 
1. Clone the repository (if you haven’t already):  
   ```bash
   git clone <repository-url>
   cd RAG_App
   ```
2. Create and activate a virtual environment:
   On Windows
   ```bash
   python -m venv venv  
   source venv/bin/activate
   ```
   On macOS/Linux
   ```bash
   python -m venv venv  
   source venv/bin/activate
   ```
3. Install dependencies, run:
```bash
pip install -r requirements.txt
```

   

## Usage
1. **Data Preparation**: Ensure that the essay is available in the `paul_graham_essay.txt` file.
2. **Streamlit Interface**: Launch the Streamlit interface for interactive analysis.
```bash
streamlit run app.py
```

## Directory Structure
```
RAG/
├── paul_graham_essay.txt      # File containing Paul Graham's essay
├── app.py                     # Streamlit application file
├── README.md                  # This README file
└── requirements.txt           # List of dependencies
```

## Technology Stack
- **LangChain**: For building the retrieval and generation pipelines.
- **Gemini-1.5-pro**: For efficient and scalable text processing.
- **Streamlit**: For creating an interactive web interface.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License.
