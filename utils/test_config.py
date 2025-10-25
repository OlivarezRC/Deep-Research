# Configuration testing utility for BSP AI Assistant
# This script validates and displays the LLM configuration file
# Used for debugging and verifying model configurations

import json
import os


def test_config_file():
    """
    Test and display the LLM configuration file contents.
    
    Reads the LLM configuration file and prints its contents in a formatted way.
    Handles file not found, JSON parsing errors, and other exceptions gracefully.
    
    Returns:
        None: Prints results to stdout
    """
    # Path to the sample LLM config file
    config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'llm_config/llm_config.json')

    try:
        # Read JSON data from file
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        
        # Print the formatted JSON data
        print(json.dumps(config_data, ensure_ascii=False))

    except FileNotFoundError:
        print(f"Error: Could not find the config file at {config_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in config file - {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")


# Run the test if executed directly
if __name__ == "__main__":
    test_config_file()
