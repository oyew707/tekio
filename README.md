<!-- PROJECT LOGO -->
<br />
<div align="center">

   <h3 align="center">Tekio: Self-Learning Agent System</h3>
   <p align="center">
     Tekio is an advanced, asynchronous AI agent framework designed to execute complex, multi-step reasoning tasks by operating within a simulated browser environment. It employs advanced planning, execution, and reflection loops to achieve high-fidelity task completion.
     <br /><br />
     <a href="https://github.com/oyew707/tekio"><strong style="color: #4CAF50;">Explore the codebase »</strong></a>
     <br /><br />
     <a href="https://github.com/oyew707/tekio">View Live Demo</a>
     &middot;     <a href="https://github.com/oyew707/tekio/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
     &middot;     <a href="https://github.com/oyew707/tekio/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
   </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#architecture">System Architecture</a></li>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation & Setup</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage Flow</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
[![Tekio Live Demo](https://img.shields.io/badge/Tekio-Live%20Demo-blue?style=flat-square)](YOUR_DEMO_LINK)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](YOUR_STATUS_LINK)

Tekio is an advanced, asynchronous AI agent framework designed to execute complex, multi-step reasoning tasks by operating within a simulated browser environment. It employs advanced planning, execution, and reflection loops to achieve high-fidelity task completion.

### System Architecture
The system is highly decoupled and distributed across several services:
*   **Agent Node (`agent/node.py`):** The core reasoning engine orchestrating tool use, planning, and execution. It implements a **Partially Observable Markov Decision Process (POMDP)** framework, maintaining a structured **Belief State** to track environment hypotheses and task progress. This allows for robust decision-making and efficient context management via a Markovian message window. This is powered by advanced capabilities mapped from **Fara 1.5B** and controls actions via **Playwright MCP tools**.
*   **Orchestration Layer:** The reasoning process utilizes **LangChain** and **LangGraph** for sophisticated, stateful orchestration across multiple thought steps.
*   **Worker Services (`workers/`):** Dedicated microservices handle computationally expensive background tasks. The trajectory processing module employs advanced learning mechanisms inspired by the work in **arXiv:2603.10600**.
*   **Frontend (`frontend/streamlit_app.py`):** The user interface responsible for task submission and result visualization, integrated with a message queuing system for asynchronous operation.

### Key Features
*   **Asynchronous Processing:** Tasks are offloaded via a robust message queue system (RabbitMQ), enabling long-running operations without timeouts.
*   **Belief-Driven Reasoning:** Employs a structured belief-tracking loop to maintain a consistent internal model of the environment, enabling effective decision-making even when the environment is partially observable.
*   **Advanced Reasoning:** Utilizes sophisticated planning algorithms powered by the integrated LLM stack.
*   **Scalability:** Designed for high availability using a containerized stack defined in `docker-compose.yml`.

<!-- GETTING STARTED -->
## Getting Started
This system is designed as a distributed application suite. Successful execution requires running multiple interconnected services.

### Prerequisites
To run Tekio, you must have access to the following services and credentials:
1. Docker contianerization: Docker is required to run the application stack
2. AI access keys/endpoints (consider local solutions like Ollama or Llama.cpp)
*   A primary Agentic VLM service (e.g., OpenAI) for high-level reasoning.
*   A dedicated LLM instance for knowledge extraction/low-level tasks .
*   An embedding model service.
3. Observability: A LangSmith account configured with the necessary API keys for tracking and debugging runs.

### Installation & Setup
1.  **Clone the Repository:**
    ```sh
    git clone https://github.com/oyew707/tekio.git
    cd tekio
    ```
2.  **Configure Environment:**
    Configure your `.env` file (or environment variables) with necessary keys, endpoint URLs, and service connection strings. For instance
    ```
    API_BASE_URL=https://chat.madebystein.dev/v1/
    API_KEY=
    OPENAI_EMBEDDING_MODEL=
    CDP_ENDPOINT=http://172.28.0.10:9222
    RABBITMQ_URL=http://rabbitmq:5672
    PGVECTOR_URL=postgresql://tekio:tekio@postgres:5432/tekio
    LANGCHAIN_TRACING_V2=true
    LANGSMITH_API_KEY=
    LANGCHAIN_PROJECT=tekio-browser-use
    VIEWPORT_SIZE=1280x720
    EXTRACTION_MODEL=
    ```
3.  **Run Services (via Docker Compose):**
    Use `docker-compose.yml` to spin up all dependent services:
    ```sh
    # Builds images, creates network, starts all containers
    docker compose up -d --build 
    ```
4. **Access Points:** Once containers are running, you can interact with the services via:
* **Frontend Application**: http://localhost:8501
* **Message Queue Management**: http://localhost:15672


<!-- USAGE -->
## Usage Flow
1.  **Submission:** The user interacts with the `Tekio` application via the Frontend.
2.  **Queuing:** When a task requires deep processing, the Frontend publishes a task object to the message queue.
3.  **Execution:** The `queue_consumer` picks up the task and passes it to the appropriate agent/worker chain.
4.  **Completion:** The result is published back through the queue, eventually being picked up by the Frontend for display.


<!-- CONTACT -->
## Contact
Stein Oyewole - eo2233@nyu.edu
Project Link: [https://github.com/oyew707/tekio](https://github.com/oyew707/tekio)