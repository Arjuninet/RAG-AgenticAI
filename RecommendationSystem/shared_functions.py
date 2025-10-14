import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import json
import re
from typing import Dict,Any,List,Optional

client = chromadb.Client()

def load_food_data(file_path: str) -> List[Dict]:
    """Load food data from JSON and the JSON data is"""
    try:
            
        with open(file_path, 'r', encoding='utf-8') as file:
            food_data = json.load(file)
        #Ensure each item has required fields and normalize the structure
        for i,item in enumerate(food_data):
            if 'food_id' not in item:
                item['food_id'] =str(i+1)
            else:
                item['food_id'] = str(item['food_id'])
            
            if 'food_ingredients' not in item:
                item['food_ingredients'] = []
            if 'food_description' not in item:
                item['food_description'] = ''
            if 'cuisine_type' not in item:
                item['cuisine_type'] = 'Unknown'
            if 'food_calories_per_serving' not in item:
                item['food_calories_per_serving'] = 0
            
            # Exatract taste features from nested food_features if available
            if 'food_features' in item and isinstance(item['food_features'], dict):
                taste_features = []
                for key, value in item['food_features'].items():
                    if value:
                        taste_features.append(str(value))
                    item['taste_profile'] = ', '.join(taste_features)
            else:
                item['taste_profile'] = ''
        print(f"Loaded successfully {len(food_data)} food items from {file_path}")
        return food_data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

def create_similarity_search_collection(collection_name: str, collection_metadata: dict = None):
    """Create ChromaDb collection with sentence transformer embdedding"""
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    #create embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    #Create new collection

    return client.create_collection(name=collection_name, embedding_function=sentence_transformer_ef, metadata=collection_metadata)

def populate_similarity_collection(collection, food_items: List[Dict]):
    """Populate collection with food data and generate embeddings"""
    documents = []
    metadatas = []
    ids = []

    #Create unique ids to avoid duplicates
    user_ids = set()

    for i,food in enumerate(food_items):
        # Create comprehensive text for embedding using rich JSON structure
        text = f"{food.get('food_name','')} - {food.get('food_description','')} Ingredients: {', '.join(food.get('food_ingredients',[]))}. Cuisine: {food.get('cuisine_type','Unknown')}. Cooking Method: {food.get('cooking_method','')}."

        # Add taste profile from food_features
        taste_profile = food.get('taste_profile','')
        if taste_profile:
            text += f" Taste Profile and features: {taste_profile}."
        
        # Add food bebefits if available
        health_benefits = food.get('food_health_benefits','')
        if health_benefits:
            text += f" Health Benefits: {health_benefits}."
        
        #Add nutritional info if available
        if 'food_nutritional_factors' in food:
            nutrition = food['food_nutritional_factors']
            if isinstance(nutrition, dict):
                nutrition_info = ', '.join([f"{k}: {v}" for k,v in nutrition.items() if v])
                if nutrition_info:
                    text += f"Nutrition: {nutrition_info}."

        #Generate Unique ID to avoid duplicates
        base_id = str(food.get('food_id', i))
        unique_id = base_id
        counter = 1
        while unique_id in user_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        user_ids.add(unique_id)
        
        documents.append(text)
        ids.append(unique_id)
        metadatas.append({
            "name": food["food_name"],
            "cuisine_type": food.get("cuisine_type", "Unknown"),
            "ingredients": ", ".join(food.get("food_ingredients", [])),
            "calories": food.get("food_calories_per_serving", 0),
            "description": food.get("food_description", ""),
            "cooking_method": food.get("cooking_method", ""),
            "health_benefits": food.get("food_health_benefits", ""),
            "taste_profile": food.get("taste_profile", "")
        })

        #Add all data to collection
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Added {len(documents)} items to the collection '{collection.name}'")

def perform_similarity_search(collection, query: str, n_results: int = 5) -> List[Dict]:
    """Perform similarity search and return formatted results"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return []
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (1 - distance)
            similarity_score = 1 - results['distances'][0][i]
            
            result = {
                'food_id': results['ids'][0][i],
                'food_name': results['metadatas'][0][i]['name'],
                'food_description': results['metadatas'][0][i]['description'],
                'cuisine_type': results['metadatas'][0][i]['cuisine_type'],
                'food_calories_per_serving': results['metadatas'][0][i]['calories'],
                'similarity_score': similarity_score,
                'distance': results['distances'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
        
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []   
