import requests
import time

# Define the API endpoint
API_URL = "http://127.0.0.1:5001/query"

def send_query(query):
    """
    Sends a query to the Flask API and returns the response.
    
    Args:
        query (str): The question or query to send.
    
    Returns:
        str: The response from the API.
    """
    # Prepare the payload
    payload = {"query": query}
    
    try:
        # Send the POST request
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        
        # Parse the JSON response
        data = response.json()
        return data.get("response", "No response received.")
    
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Welcome to the Query Interface!")
    print("Type your query below (or type 'exit' to quit):")
    
    while True:
        # Get user input
        query = input("Your Query: ")
        
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        # Send the query and print the response

        start_time = time.time()
        
        response = send_query(query)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

        print("Response:", response)
