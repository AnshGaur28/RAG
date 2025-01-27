# RAG: AI-based Chatbot

An AI-based chatbot leveraging Langchain, OpenAI, and vector databases to answer queries related to a data store. This chatbot is easily deployable to websites and seamlessly adaptive to user needs. It is customizable to user data and requirements.

## Technology Stack

1. **Frontend**: Swagger GUI
2. **Backend**: Python, Langchain, OpenAI
3. **Database**: Weaviate, Mongoose

## How to Use

1. **Set up an account** on [Weaviate](https://weaviate.io/) and [OpenAI](https://www.openai.com/).
2. **Generate an OpenAI embedding API key**.
3. **Connect to the Weaviate cloud database** using the URL and API key of your Weaviate cluster.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Weaviate account
- OpenAI account

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/RAG-AI-Chatbot.git
    cd RAG-AI-Chatbot
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1. Set up environment variables:
    ```sh
    export OPENAI_API_KEY='your_openai_api_key'
    export WEAVIATE_API_URL='your_weaviate_api_url'
    export WEAVIATE_API_KEY='your_weaviate_api_key'
    ```

### Running the Application

1. Start the backend server:
    ```sh
    python app.py
    ```

2. Access the Swagger GUI to interact with the API.

## Customization

The chatbot can be customized to fit specific user data and requirements. Follow these steps:

1. Modify the `data_loader.py` script to load your custom data into the Weaviate database.
2. Update the `query_handler.py` to handle your specific query logic.

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.



## Contact

For questions or support, please reach out to [ansh28.dinesh30@gmail.com]

