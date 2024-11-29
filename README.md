# BLOOM-based Query System

This project implements a query system using the BLOOM language model and FAISS for efficient document retrieval. The system is designed to handle natural language queries and provide relevant responses based on a pre-indexed set of documents.

## Setup and Installation

### Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- Python 3.8

### Building the Docker Image

1. Clone the repository:
    ```sh
    git clone https://github.com/CcEnrico/bloom_testing.git
    cd bloom_testing
    ```

2. Build the Docker image:
    ```sh
    docker build -t bloom-query-system .
    ```

3. Run the Docker container:
    ```sh
    docker run --gpus all -p 5000:5000 bloom-query-system
    ```

## Usage

### Authentication with Hugging Face

To use the BLOOM model, you need to authenticate with your Hugging Face token. Follow these steps to set up authentication:

1. Obtain your Hugging Face token from [Hugging Face](https://huggingface.co/settings/tokens).

2. Set the token as an environment variable:
    ```sh
    export HUGGINGFACE_TOKEN=your_hugging_face_token
    ```

3. Verify the token is set correctly:
    ```sh
    echo $HUGGINGFACE_TOKEN
    ```

Replace `your_hugging_face_token` with the token you obtained from Hugging Face.

### Logging into Hugging Face via Terminal

To log in to Hugging Face via the terminal, use the following command:

```sh
huggingface-cli login
```

You will be prompted to enter your Hugging Face token. Once authenticated, you can use the BLOOM model in your application.

### test local

To test the system locally without Docker, you can set up a Python virtual environment and install the required dependencies.

1. Create a virtual environment:
    ```sh
    python3 -m venv bloom
    ```

2. Activate the virtual environment:
    ```sh
    source bloom/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the application:
    ```sh
    python app.py
    ```

### Preparing Resources

The resources (models, tokenizers, FAISS index, and document embeddings) are prepared during the Docker build process. The `resources.py` script handles the preparation and loading of these resources.

### Running the Flask API

The Flask API is started automatically when the Docker container is run. It listens on port 5000 for incoming queries.

### Sending Queries

You can interact with the system using the `interact.py` script:

```sh
python interact.py
```