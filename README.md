# SQL Tutor with LangChain and RAG

This project is a personal learning endeavor to deepen my understanding of SQL while exploring powerful tools like LangChain and Retrieval-Augmented Generation (RAG). 
My goal was to build a contextual chatbot that acts as a personal study assistant, answering questions about SQL notes from my Big Data course at Emory. 
The application is designed to be modular, efficient, and leverages modern package management and containerization practices.

### Key Technologies

* **LangChain:** A python framework used for orchestrating the LLM and its interactions with external data using a ConversationalRetrievalChain powered by the gpt-3.5-turbo model.
* **Pinecone:** A vector database for efficient storage and retrieval of document embeddings.
* **RAG (Retrieval-Augmented Generation):** The core technique that allows the chatbot to retrieve information from my SQL notes before generating a response.
* **Docker:** For creating a consistent and isolated environment for the entire application, including all dependencies.
* **uv:** A fast and modern Python package installer.
* **Streamlit:** For creating a simple, user-friendly interface for the chatbot.

### Live Demo

View the App Demo! 
<video src="https://raw.githubusercontent.com/amndongozi/SQL-RAG-Pipeline/main/images/LiveDemo-SQLTutor-ezgif.com-video-cutter.mp4" controls autoplay loop muted></video>

### How to Run the Project

This project is fully containerized with Docker, ensuring a consistent and isolated environment for all dependencies and tools.

#### **Step 1: Set up the Environment**

1.  Clone the repository and navigate to the project directory.
    ```bash
    git clone [https://github.com/amndongozi/SQL-RAG-Pipeline.git]
    cd SQL-RAG-Pipeline
    ```

2.  Ensure you have Docker installed and running on your system.

3.  Build the Docker image. The name of the image will be `langchain-app`:
    ```bash
    docker build -t langchain-app .
    ```

4.  Create a `.env` file in the root directory and add your environment variables (e.g., Pinecone API keys). This file's contents will be securely exposed to the Docker container.

#### **Step 2: Run the Chatbot**

1.  Execute the following command to run the container, exposing the necessary ports, and mounting your local project directory as a volume.
    ```bash
    docker run --env-file .env -p 8888:8888 -p 8501:8501 -v C:/Users/ndong/PyCharmProjects/PythonLangchain:/app langchain-app
    ```

2.  From the terminal, run the Streamlit app:
    ```bash
    streamlit run Chatbot.py
    ```
3.  Once the notebook setup is complete and the Streamlit app is running, access the chatbot application from your web browser by navigating to `http://localhost:8501`.
