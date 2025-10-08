import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
    )

#create cromadb instance
client = chromadb.Client()
#Define the name for the collection to be created
collection_name = "my_grocery_collection"

def main():
    try:
        collection = client.create_collection(
            name=collection_name,
            metadata = {"description":"A collection for storing grocery data"},
            configuration={
                "hnsw":{"space":"cosine"},
                "embedding_function":ef
            }
        )
        print(f"Collection created: {collection.name}")
        # Array of grocery-related text items
        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]
        
        #Add documents and thier corresponding ID's to the collection
        collection.add(
            documents=texts,
            metadatas=[{"source":"grocery_store","category":"food"} for _ in texts],
            ids=ids
        )
        #Get all the item from collection
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        #Function to perform a similarity search in the collection
        def perform_similarity_search(collection, all_items):
            try:
                query_term = "apple"
                #Perform a query to search for the most similar documents.
                results = collection.query(
                    query_texts=[query_term],
                    n_results=3 #Retrieve top 3 results
                )
                print(f"Query results for '{query_term}':")
                print(results)

                #Check if no result are retured or if the result are none
                if not results or not results['ids'] or len(results['ids'][0])==0:
                    #Log a message indicating that no similar documents
                    print(f"No documents found similat to {query_term}")
                    return

                print(f'Top 3 similar documents to "{query_term}"')

                for i in range(min(3, len(results['ids'][0]))):
                    doc_id = results['ids'][0][i]  # Get ID from 'ids' array
                    score = results['distances'][0][i] #Get distance score
                    #Retrieve text data from the results
                    text = results['documents'][0][i]

                    if not text:
                        print(' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                    else:
                        print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
            except Exception as e:
                print(f"Error in similarity search: {e}")

        perform_similarity_search(collection, all_items)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
