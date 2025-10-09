from shared_functions import *

#global variable to store loaded food items
food_items = []

def interactive_food_chatbot(collection):
    """Interactive chatbot for food recommendations."""
    print("\n" + "=" * 50)
    print("INTERACTIVE FOOD SEARCH CHATBOT")
    print("=" * 50)
    print("Commands:")
    print("  • Type any food name or description to search")
    print("  • 'help' - Show available commands")
    print("  • 'quit' or 'exit' - Exit the system")
    print("  • Ctrl+C - Emergency exit")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n Search for food: ").strip()

            #handle empty input
            if not user_input:
                print("Please enter a search term or 'help' for commands.")
                continue

            #Handle exit commands
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting the system. Goodbye!")
                break

            #handle help command
            elif user_input.lower() == 'help':
                show_help_menu()
            
            else:
                handle_food_search(collection,user_input)

        except KeyboardInterrupt:
            print("\nExiting the system. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

def show_help_menu():
    """Display help information for users"""
    print("\n📖 HELP MENU")
    print("-" * 30)
    print("Search Examples:")
    print("  • 'chocolate dessert' - Find chocolate desserts")
    print("  • 'Italian food' - Find Italian cuisine")
    print("  • 'sweet treats' - Find sweet desserts")
    print("  • 'baked goods' - Find baked items")
    print("  • 'low calorie' - Find lower-calorie options")
    print("\nCommands:")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the system")

def handle_food_search(collection, query):
    """Handle food similarity search with enhanced display"""
    print(f"\n🔍 Searching for '{query}'...")
    print("   Please wait...")
    
    # Perform similarity search
    results = perform_similarity_search(collection, query, 3)
    
    if not results:
        print("❌ No matching foods found.")
        print("💡 Try different keywords like:")
        print("   • Cuisine types: 'Italian', 'American'")
        print("   • Ingredients: 'chocolate', 'flour', 'cheese'")
        print("   • Descriptors: 'sweet', 'baked', 'dessert'")
        return
    
    # Display results with rich formatting
    print(f"\n✅ Found {len(results)} recommendations:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        # Calculate percentage score
        percentage_score = result['similarity_score'] * 100
        
        print(f"\n{i}. 🍽️  {result['food_name']}")
        print(f"   📊 Match Score: {percentage_score:.1f}%")
        print(f"   🏷️  Cuisine: {result['cuisine_type']}")
        print(f"   🔥 Calories: {result['food_calories_per_serving']} per serving")
        print(f"   📝 Description: {result['food_description']}")
        
        # Add visual separator
        if i < len(results):
            print("   " + "-" * 50)
    
    print("=" * 60)
    
    # Provide suggestions for further exploration
    suggest_related_searches(results)

def suggest_related_searches(results):
    """Suggest related searches based on current results"""
    if not results:
        return
    
    # Extract cuisine types from results
    cuisines = list(set([r['cuisine_type'] for r in results]))
    
    print("\n💡 Related searches you might like:")
    for cuisine in cuisines[:3]:  # Limit to 3 suggestions
        print(f"   • Try '{cuisine} dishes' for more {cuisine} options")
    
    # Suggest calorie-based searches
    avg_calories = sum([r['food_calories_per_serving'] for r in results]) / len(results)
    if avg_calories > 350:
        print("   • Try 'low calorie' for lighter options")
    else:
        print("   • Try 'hearty meal' for more substantial dishes")


def main():
    """Main function for interactive CLI food recommendation system."""
    try:
        print("Welcome to the Interactive Food Recommendation System!")
        print("=" * 50)
        print("Loading food database...")

        #Load food data from file
        global food_items
        food_items = load_food_data('RecommendationSystem/FoodDataSet.json')
        print(f"Loaded {len(food_items)} food items.")

        #Create and populate search collection
        collection = create_similarity_search_collection(
            "interactive_food_collection",
            {'description':'A collection for interactive food search'}
        )
        populate_similarity_collection(collection, food_items)

        #Start interactive chatbot
        interactive_food_chatbot(collection)
    except Exception as e:
        print(f"An error initializing system: {e}")

if __name__ == "__main__":
    main()