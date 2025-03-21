#!/usr/bin/env python3

import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import base64
from tqdm import tqdm
import re

# Load environment variables
load_dotenv()

# Configure logging to show only role, content, and total tokens
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaypointPredictionExperiment:
    """
    Experiment to test different vision language models' ability to predict
    3D coordinates of items based on an image and known item coordinates.
    """
    
    def __init__(
        self,
        image_path: str,
        item_location_yaml: str,
        models: List[str],
        num_predictions: int,
        items_to_predict: List[str],
        api_key: Optional[str] = None,
        output_dir: str = "experiment_results",
        prompt_template: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the experiment.
        
        Args:
            image_path: Path to the image showing the table with items
            item_location_yaml: Path to the YAML file with known item coordinates
            models: List of model names to test
            num_predictions: Number of predictions to make per model
            items_to_predict: List of items whose coordinates need to be predicted
            api_key: API key for OpenRouter (defaults to env var)
            output_dir: Directory to save experiment results
            prompt_template: Custom prompt template to use (default is None, which uses the standard prompt)
            experiment_name: Optional name to identify this experiment run in output files
        """
        self.image_path = image_path
        self.item_location_yaml = item_location_yaml
        self.models = models
        self.num_predictions = num_predictions
        self.items_to_predict = items_to_predict
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.output_dir = output_dir
        self.prompt_template = prompt_template
        self.experiment_name = experiment_name
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create directory for raw responses
        self.raw_response_dir = os.path.join(output_dir, "raw_responses")
        os.makedirs(self.raw_response_dir, exist_ok=True)
        
        # Will store all known item locations
        self.known_items = {}
        # Will store all prediction results
        self.results = {}
        # Store raw responses for debugging
        self.raw_responses = {}
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check if YAML file exists
        if not os.path.exists(item_location_yaml):
            raise FileNotFoundError(f"Item location YAML file not found: {item_location_yaml}")
            
        # Check if API key is available
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set the OPENROUTER_API_KEY environment variable.")
    
    def load_item_locations(self) -> Dict[str, List[float]]:
        """
        Load known item locations from YAML file.
        
        Returns:
            Dictionary mapping item names to their 3D coordinates
        """
        try:
            with open(self.item_location_yaml, 'r') as f:
                data = yaml.safe_load(f)
                
            if not data or 'items' not in data:
                raise ValueError("Invalid YAML format: 'items' key not found")
                
            item_locations = {}
            
            # Process each item
            for item_name, item_data in data['items'].items():
                if 'coordinates' in item_data and all(coord is not None for coord in item_data['coordinates']):
                    item_locations[item_name] = item_data['coordinates']
            
            self.known_items = item_locations
            
            # Validate items to predict
            unknown_items = [item for item in self.items_to_predict if item not in data['items']]
            if unknown_items:
                logger.warning(f"The following items are not in the YAML file: {unknown_items}")
                
            # Check which items have missing coordinates
            items_without_coords = []
            for item in self.items_to_predict:
                if item in data['items'] and (
                    'coordinates' not in data['items'][item] or 
                    any(coord is None for coord in data['items'][item].get('coordinates', [None]))
                ):
                    items_without_coords.append(item)
            
            logger.info(f"Loaded {len(item_locations)} known item locations")
            logger.info(f"Items to predict: {self.items_to_predict}")
            
            return item_locations
        
        except Exception as e:
            logger.error(f"Error loading item locations: {e}")
            raise
    
    def encode_image(self) -> str:
        """
        Encode the image as base64 for inclusion in API requests.
        
        Returns:
            Base64-encoded image as a data URI
        """
        with open(self.image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"
    
    def query_model(self, model_name: str, item_name: str) -> Tuple[bool, Optional[List[float]], Optional[str]]:
        """
        Query a single model to predict the coordinates of a specific item.
        
        Args:
            model_name: Name of the vision language model
            item_name: Name of the item to predict coordinates for
            
        Returns:
            Tuple of (success, coordinates, explanation)
        """
        # Encode image
        image_data_uri = self.encode_image()
        
        # Construct the prompt
        known_items_text = ""
        for name, coords in self.known_items.items():
            # Skip the item we're trying to predict
            if name == item_name:
                continue
            known_items_text += f"- {name}: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}] meters from robot base\n"
        
        # Use custom prompt if provided, otherwise use default
        if self.prompt_template:
            user_prompt = self.prompt_template.format(
                known_items_text=known_items_text,
                item_name=item_name
            )
        else:
            user_prompt = f"""I have a table with various items. I know the 3D coordinates (in meters) of some items, measured from the robot base:

{known_items_text}

Based on the image and the known item positions, predict the 3D coordinates of the {item_name} in the same coordinate system.

Return your answer as a JSON object with this exact format (fields can be in any order):
{{
    "item": "{item_name}",
    "coordinates": [x, y, z],
    "explanation": "your detailed reasoning"
}}

Use only numbers (not strings) for the coordinates, with no units. Express coordinates in meters.
"""
        
        system_prompt = f"""You are an expert in spatial reasoning and computer vision. You can accurately predict the 3D position of objects in space given visual information and reference coordinates.

The coordinate system is in meters with the origin at the robot base. Positive X is forward, positive Y is left, and positive Z is up.

Analyze the image carefully, and use the known item positions as reference points to triangulate and estimate the position of the requested item.
"""
        
        # Construct the messages payload
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ]
            }
        ]
        
        # Set up API request
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/FYP-kitchen-assistive-robot",
            "X-Title": "Kitchen Assistive Robot Waypoint Prediction",
        }
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 500
        }
        
        try:
            # Make the request
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            
            # Save the raw response for debugging even if there's an error
            timestamp = int(time.time())
            raw_response_file = os.path.join(
                self.raw_response_dir, 
                f"raw_response_{model_name.replace('/', '_')}_{item_name}_{timestamp}.json"
            )
            
            try:
                # Try to save the JSON response
                with open(raw_response_file, 'w') as f:
                    json.dump(response.json(), f, indent=2)
            except:
                # If that fails, save the raw text
                with open(raw_response_file, 'w') as f:
                    f.write(response.text)
                    
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Store raw response in memory
            if model_name not in self.raw_responses:
                self.raw_responses[model_name] = {}
            if item_name not in self.raw_responses[model_name]:
                self.raw_responses[model_name][item_name] = []
            self.raw_responses[model_name][item_name].append(response_data)
            
            # Get the response content
            if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"].strip()
                token_usage = response_data["usage"]["total_tokens"] if "usage" in response_data else "n/a"
                
                # Log only role, content, and total tokens
                log_data = {
                    "role": "assistant",
                    "content": content[:100] + "..." if len(content) > 100 else content,
                    "total_tokens": token_usage
                }
                logger.info(f"Model response: {json.dumps(log_data, indent=2)}")
                
                # Log the full response for debugging
                logger.info(f"Full response content: {content}")
                
                # Also save the processed content directly
                content_file = os.path.join(
                    self.raw_response_dir, 
                    f"content_{model_name.replace('/', '_')}_{item_name}_{timestamp}.txt"
                )
                with open(content_file, 'w') as f:
                    f.write(content)
                
                # Extract the JSON from the response
                try:
                    # Find JSON in the response - it might be wrapped in code blocks
                    json_match = None
                    if '```json' in content:
                        parts = content.split('```json')
                        if len(parts) > 1:
                            json_text = parts[1].split('```')[0].strip()
                            json_match = json_text
                    elif '```' in content:
                        parts = content.split('```')
                        if len(parts) > 1:
                            json_text = parts[1].strip()
                            json_match = json_text
                    else:
                        # Try the whole content
                        json_match = content
                    
                    if json_match:
                        try:
                            # First try to parse as proper JSON
                            result = json.loads(json_match)
                            
                            # Log the exact structure received for debugging
                            logger.info(f"Parsed JSON structure: {result}")
                            
                            coordinates = result.get("coordinates", None)
                            explanation = result.get("explanation", "No explanation provided")
                            
                            # Validate coordinates format
                            valid_coordinates = (coordinates is not None and 
                                               isinstance(coordinates, list) and 
                                               len(coordinates) == 3 and 
                                               all(isinstance(c, (int, float)) for c in coordinates))
                            
                            if not valid_coordinates:
                                logger.warning(f"Invalid coordinates in JSON response: {coordinates}")
                                
                                # Special handling for common errors
                                if isinstance(coordinates, str):
                                    # Sometimes models return coordinates as a string like "[x, y, z]"
                                    try:
                                        import re
                                        coords_match = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', coordinates)
                                        if coords_match:
                                            x, y, z = float(coords_match.group(1)), float(coords_match.group(2)), float(coords_match.group(3))
                                            coordinates = [x, y, z]
                                            logger.info(f"Successfully extracted coordinates from string: {coordinates}")
                                            return True, coordinates, explanation
                                    except Exception as e:
                                        logger.error(f"Error extracting coordinates from string: {e}")
                                
                                # Try to extract coordinates from explanation if they exist there
                                if explanation and isinstance(explanation, str):
                                    logger.info("Attempting to extract coordinates from explanation")
                                    
                                    import re
                                    coords_match = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', explanation)
                                    if coords_match:
                                        try:
                                            x, y, z = float(coords_match.group(1)), float(coords_match.group(2)), float(coords_match.group(3))
                                            coordinates = [x, y, z]
                                            logger.info(f"Successfully extracted coordinates from explanation: {coordinates}")
                                            # Update explanation to remove the coordinates to avoid duplication
                                            explanation = explanation.replace(coords_match.group(0), "")
                                            return True, coordinates, explanation
                                        except Exception as e:
                                            logger.error(f"Error extracting coordinates from explanation: {e}")
                                
                                # If all else fails, try the fallback method
                                return self.query_model_fallback(content, item_name)
                            
                            logger.info(f"Successfully extracted valid coordinates: {coordinates}")
                            return True, coordinates, explanation
                        except (json.JSONDecodeError, ValueError) as e:
                            # If JSON parsing fails, try to extract coordinates from text
                            logger.warning(f"JSON parsing failed: {e}. Attempting to extract coordinates from text")
                            # Try to extract coordinates as a fallback
                            return self.query_model_fallback(content, item_name)
                    else:
                        logger.warning("No JSON found in response")
                        return self.query_model_fallback(content, item_name)
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from response: {e}")
                    # Try to extract coordinates as a fallback
                    return self.query_model_fallback(content, item_name)
            else:
                logger.error(f"Invalid response structure: {response_data}")
                return False, None, None
            
        except Exception as e:
            logger.error(f"Error querying model {model_name}: {e}")
            return False, None, None
    
    def query_model_fallback(self, content: str, item_name: str) -> Tuple[bool, Optional[List[float]], Optional[str]]:
        """
        Fallback method to extract coordinates from text content when JSON parsing fails.
        
        Args:
            content: The text content to extract coordinates from
            item_name: The name of the item being predicted
            
        Returns:
            Tuple of (success, coordinates, explanation)
        """
        logger.info(f"Entering fallback extraction for content: {content[:100]}...")
        
        # Look for patterns like [x, y, z] or (x, y, z) or x, y, z
        coord_patterns = [
            # r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',  # [x, y, z]
            # r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',  # (x, y, z)
            # r'coordinates:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',  # coordinates: [x, y, z]
            # r'coordinates\s*=\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',  # coordinates = [x, y, z]
            # r'coordinates.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # coordinates ... x, y, z
            # r'position.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # position ... x, y, z
            # r'coordinate.*?x\s*=\s*(-?\d+\.?\d*).*?y\s*=\s*(-?\d+\.?\d*).*?z\s*=\s*(-?\d+\.?\d*)',  # x = x, y = y, z = z
            # r'X\s*=\s*(-?\d+\.?\d*).*?Y\s*=\s*(-?\d+\.?\d*).*?Z\s*=\s*(-?\d+\.?\d*)',  # X = x, Y = y, Z = z
            # Numbers with units
            r'(-?\d+\.?\d*)\s*meters?\s*[,;]\s*(-?\d+\.?\d*)\s*meters?\s*[,;]\s*(-?\d+\.?\d*)\s*meters?',  # x meters, y meters, z meters
            # Final coordinates sentences
            r'final coordinates are\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # final coordinates
            r'my prediction is\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',      # my prediction
            r'I estimate the coordinates to be\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # estimate to be
            r'placing it at\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # placing it at
            r'predict the position to be\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # predict the position
            r'coordinates are\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # coordinates are
            r'position is\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # position is
        ]
        
        # Add item-specific patterns with the real item name
        item_specific_patterns = [
            fr'estimated position of the {item_name} is\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # Natural language
            fr'the {item_name} is located at\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # "located at" x, y, z
            fr'final coordinates for the {item_name}\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # "final coordinates"
            fr'determined the {item_name}\'s coordinates to be\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # "determined to be"
            fr'estimate for the {item_name}\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # "estimate for"
            fr'conclude that the {item_name} is at\s*.*?(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',  # "conclude that"
        ]
        coord_patterns.extend(item_specific_patterns)
        
        for pattern in coord_patterns:
            matches = re.search(pattern, content)
            if matches:
                try:
                    x, y, z = float(matches.group(1)), float(matches.group(2)), float(matches.group(3))
                    coords = [x, y, z]
                    logger.info(f"Successfully extracted coordinates from text with pattern '{pattern}': {coords}")
                    
                    # Extract explanation - take the whole response as the explanation
                    explanation = content
                    # Remove any detected coordinates to avoid duplication in the explanation
                    explanation = explanation.replace(matches.group(0), "")
                    
                    return True, coords, explanation
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to convert extracted coordinates to float using pattern '{pattern}': {e}")
                    continue
        
        # For last resort - just look for 3 numbers in sequence
        simple_pattern = r'(?<!\d)(-?\d+\.?\d*)(?!\d)\D+(?<!\d)(-?\d+\.?\d*)(?!\d)\D+(?<!\d)(-?\d+\.?\d*)(?!\d)'
        matches = re.search(simple_pattern, content)
        if matches:
            try:
                x, y, z = float(matches.group(1)), float(matches.group(2)), float(matches.group(3))
                coords = [x, y, z]
                logger.info(f"Last resort: found 3 numbers in sequence that might be coordinates: {coords}")
                return True, coords, content
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to extract coordinates with simple pattern: {e}")
        
        logger.warning("Could not extract coordinates from text")
        return False, None, content  # Return the content as the explanation even without coordinates
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the full experiment.
        
        Returns:
            Dictionary with all experiment results
        """
        # Load known item locations
        self.load_item_locations()
        
        # Initialize results structure
        results = {
            "experiment_config": {
                "image_path": self.image_path,
                "item_location_yaml": self.item_location_yaml,
                "models": self.models,
                "num_predictions": self.num_predictions,
                "items_to_predict": self.items_to_predict,
                "prompt_template": self.prompt_template
            },
            "predictions": {},
            "metrics": {}
        }
        
        # For each model
        for model_name in self.models:
            logger.info(f"Running predictions with model: {model_name}")
            
            model_predictions = {}
            
            # For each item to predict
            for item_name in self.items_to_predict:
                logger.info(f"  Predicting coordinates for: {item_name}")
                
                item_predictions = []
                item_explanations = []
                
                # Make n predictions
                for i in tqdm(range(self.num_predictions), desc=f"{model_name} - {item_name}"):
                    success, coordinates, explanation = self.query_model(model_name, item_name)
                    
                    if success and coordinates:
                        item_predictions.append(coordinates)
                        item_explanations.append(explanation)
                    elif explanation:  # Store explanation even if coordinates weren't successfully extracted
                        item_explanations.append(f"[NO COORDINATES] {explanation[:200]}...")
                    
                    # Small delay to avoid rate limits
                    time.sleep(1)
                
                model_predictions[item_name] = {
                    "coordinates": item_predictions,
                    "explanations": item_explanations
                }
            
            results["predictions"][model_name] = model_predictions
        
        # Calculate metrics
        metrics = self.calculate_metrics(results["predictions"])
        results["metrics"] = metrics
        
        # Save results
        timestamp = int(time.time())
        if self.experiment_name:
            result_file = os.path.join(self.output_dir, f"experiment_results_{self.experiment_name}_{timestamp}.json")
        else:
            result_file = os.path.join(self.output_dir, f"experiment_results_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment results saved to: {result_file}")
        
        # Generate visualization
        self.visualize_results(results, timestamp)
        
        self.results = results
        return results
    
    def calculate_metrics(self, predictions: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for all predictions.
        
        Args:
            predictions: Dictionary of all predictions by model and item
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        for model_name, model_preds in predictions.items():
            model_metrics = {}
            
            for item_name, item_data in model_preds.items():
                item_preds = item_data["coordinates"]
                
                # Skip if no predictions
                if not item_preds:
                    model_metrics[item_name] = {
                        "num_predictions": 0,
                        "mean_position": None,
                        "std_dev": None,
                        "variance": None,
                        "mean_euclidean_deviation": None,
                        "ground_truth": None,
                        "ground_truth_deviation": None
                    }
                    continue
                
                # Convert to numpy array for calculations
                preds_array = np.array(item_preds)
                
                # Calculate mean position
                mean_position = np.mean(preds_array, axis=0).tolist()
                
                # Calculate standard deviation
                std_dev = np.std(preds_array, axis=0).tolist()
                
                # Calculate variance
                variance = np.var(preds_array, axis=0).tolist()
                
                # Calculate Euclidean distances from each prediction to the mean
                distances = [np.linalg.norm(np.array(pred) - np.array(mean_position)) for pred in item_preds]
                mean_euclidean_deviation = np.mean(distances)
                
                # Get ground truth from YAML if available and calculate deviation
                ground_truth = None
                ground_truth_deviation = None
                if item_name in self.known_items:
                    ground_truth = self.known_items[item_name]
                    # Calculate Euclidean distance between mean prediction and ground truth
                    ground_truth_deviation = np.linalg.norm(np.array(mean_position) - np.array(ground_truth))
                    # Calculate individual deviations from ground truth
                    individual_deviations = [np.linalg.norm(np.array(pred) - np.array(ground_truth)) for pred in item_preds]
                    mean_ground_truth_deviation = np.mean(individual_deviations)
                
                model_metrics[item_name] = {
                    "num_predictions": len(item_preds),
                    "mean_position": mean_position,
                    "std_dev": std_dev,
                    "variance": variance,
                    "mean_euclidean_deviation": mean_euclidean_deviation,
                    "ground_truth": ground_truth,
                    "ground_truth_deviation": ground_truth_deviation,
                    "mean_ground_truth_deviation": mean_ground_truth_deviation if ground_truth else None
                }
            
            metrics[model_name] = model_metrics
        
        return metrics
    
    def visualize_results(self, results: Dict[str, Any], timestamp: int) -> None:
        """
        Generate visualizations of the prediction results.
        
        Args:
            results: Dictionary with all experiment results
            timestamp: Timestamp for the filename
        """
        for item_name in self.items_to_predict:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot known items
            for known_item, coords in self.known_items.items():
                # Highlight the ground truth for target item if available
                if known_item == item_name:
                    ax.scatter(coords[0], coords[1], coords[2], label=f"Ground Truth: {known_item}", 
                               marker='o', s=200, color='green', edgecolors='black')
                else:
                    ax.scatter(coords[0], coords[1], coords[2], label=f"Known: {known_item}", marker='o', s=100)
            
            # Different markers for each model
            markers = ['x', '+', '*', 'D', 'v', '^', '<', '>', 's', 'p']
            
            # Plot predictions for each model
            for i, model_name in enumerate(self.models):
                model_short_name = model_name.split('/')[-1]  # Simplify model name for legend
                
                # Get predictions for this item from this model
                if (model_name in results["predictions"] and 
                    item_name in results["predictions"][model_name] and 
                    "coordinates" in results["predictions"][model_name][item_name]):
                    
                    predictions = results["predictions"][model_name][item_name]["coordinates"]
                    
                    if predictions:
                        # Plot each prediction
                        for pred in predictions:
                            ax.scatter(pred[0], pred[1], pred[2], marker=markers[i % len(markers)], alpha=0.5)
                        
                        # Plot mean position
                        if model_name in results["metrics"] and item_name in results["metrics"][model_name]:
                            mean_pos = results["metrics"][model_name][item_name]["mean_position"]
                            if mean_pos:
                                ax.scatter(mean_pos[0], mean_pos[1], mean_pos[2], 
                                          marker=markers[i % len(markers)], s=100, 
                                          label=f"{model_short_name} (mean)", edgecolors='black')
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Predictions for {item_name}')
            
            # Add legend
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            if self.experiment_name:
                fig_path = os.path.join(self.output_dir, f"visualization_{self.experiment_name}_{item_name}_{timestamp}.png")
            else:
                fig_path = os.path.join(self.output_dir, f"visualization_{item_name}_{timestamp}.png")
            plt.savefig(fig_path)
            plt.close()
            
            logger.info(f"Visualization for {item_name} saved to: {fig_path}")
    
    def print_summary(self) -> None:
        """Print a summary of the experiment results."""
        if not self.results or not self.results.get("metrics"):
            logger.warning("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\nImage: {self.image_path}")
        print(f"Items with known locations: {list(self.known_items.keys())}")
        print(f"Items predicted: {self.items_to_predict}")
        print(f"Models tested: {[m.split('/')[-1] for m in self.models]}")
        print(f"Predictions per model-item pair: {self.num_predictions}")
        if self.prompt_template:
            print(f"Using custom prompt template")
        
        print("\n" + "-"*80)
        print("PREDICTION CONSISTENCY (Mean Euclidean Deviation in meters)")
        print("-"*80)
        
        # Table header
        header = "Model".ljust(30) + " | " + " | ".join([item.ljust(15) for item in self.items_to_predict])
        print(header)
        print("-" * len(header))
        
        # Table rows
        for model_name in self.models:
            model_short = model_name.split('/')[-1].ljust(30)
            row = model_short + " | "
            
            for item_name in self.items_to_predict:
                if (model_name in self.results["metrics"] and 
                    item_name in self.results["metrics"][model_name] and 
                    self.results["metrics"][model_name][item_name]["mean_euclidean_deviation"] is not None):
                    
                    deviation = self.results["metrics"][model_name][item_name]["mean_euclidean_deviation"]
                    row += f"{deviation:.4f}".ljust(15) + " | "
                else:
                    row += "N/A".ljust(15) + " | "
            
            print(row)
        
        print("\n" + "-"*80)
        print("PREDICTION ACCURACY (Ground Truth Deviation in meters)")
        print("-"*80)
        
        # Table header
        header = "Model".ljust(30) + " | " + " | ".join([item.ljust(15) for item in self.items_to_predict])
        print(header)
        print("-" * len(header))
        
        # Table rows
        for model_name in self.models:
            model_short = model_name.split('/')[-1].ljust(30)
            row = model_short + " | "
            
            for item_name in self.items_to_predict:
                if (model_name in self.results["metrics"] and 
                    item_name in self.results["metrics"][model_name] and 
                    self.results["metrics"][model_name][item_name]["ground_truth_deviation"] is not None):
                    
                    gt_deviation = self.results["metrics"][model_name][item_name]["ground_truth_deviation"]
                    row += f"{gt_deviation:.4f}".ljust(15) + " | "
                else:
                    row += "N/A".ljust(15) + " | "
            
            print(row)
        
        print("\n" + "-"*80)
        print("PREDICTED MEAN POSITIONS vs GROUND TRUTH (x, y, z in meters)")
        print("-"*80)
        
        for item_name in self.items_to_predict:
            print(f"\nItem: {item_name}")
            
            # Print ground truth if available
            if item_name in self.known_items:
                gt = self.known_items[item_name]
                gt_str = f"[{gt[0]:.4f}, {gt[1]:.4f}, {gt[2]:.4f}]"
                print(f"  Ground Truth: {gt_str}")
            
            for model_name in self.models:
                model_short = model_name.split('/')[-1]
                
                if (model_name in self.results["metrics"] and 
                    item_name in self.results["metrics"][model_name] and 
                    self.results["metrics"][model_name][item_name]["mean_position"] is not None):
                    
                    mean_pos = self.results["metrics"][model_name][item_name]["mean_position"]
                    mean_pos_str = f"[{mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}]"
                    
                    # Add deviation if ground truth is available
                    if self.results["metrics"][model_name][item_name]["ground_truth_deviation"] is not None:
                        dev = self.results["metrics"][model_name][item_name]["ground_truth_deviation"]
                        print(f"  {model_short}: {mean_pos_str} (deviation: {dev:.4f}m)")
                    else:
                        print(f"  {model_short}: {mean_pos_str}")
                else:
                    print(f"  {model_short}: N/A")
        
        print("\n" + "-"*80)
        print("EXPLANATIONS")
        print("-"*80)
        
        for item_name in self.items_to_predict:
            print(f"\nItem: {item_name}")
            
            for model_name in self.models:
                model_short = model_name.split('/')[-1]
                
                if (model_name in self.results["predictions"] and 
                    item_name in self.results["predictions"][model_name] and 
                    "explanations" in self.results["predictions"][model_name][item_name]):
                    
                    explanations = self.results["predictions"][model_name][item_name]["explanations"]
                    if explanations:
                        print(f"  {model_short}:")
                        for i, explanation in enumerate(explanations, 1):
                            print(f"    Prediction {i}: {explanation}")
                    else:
                        print(f"  {model_short}: No explanations available")
                else:
                    print(f"  {model_short}: No explanations available")
        
        print("\n" + "="*80)

def get_prompt_templates():
    """Return a dictionary of prompt templates for testing."""
    return {
        "standard": """I have a table with various items. I know the 3D coordinates (in meters) of some items, measured from the robot base:

{known_items_text}

Based on the image and the known item positions, predict the 3D coordinates of the {item_name} in the same coordinate system.

Return your answer as a JSON object with this exact format:
{{
    "item": "{item_name}",
    "coordinates": [x, y, z],
    "explanation": "your detailed reasoning"
}}

Use only numbers (not strings) for the coordinates, with no units. Express coordinates in meters.
""",
        "explanation_first": """I have a table with various items. I know the 3D coordinates (in meters) of some items, measured from the robot base:

{known_items_text}

Based on the image and the known item positions, predict the 3D coordinates of the {item_name} in the same coordinate system.

Return your answer as a JSON object with this exact format:
{{
    "item": "{item_name}",
    "explanation": "your detailed reasoning here",
    "coordinates": [x, y, z]
}}

Use only numbers (not strings) for the coordinates, with no units. Express coordinates in meters.
First provide your detailed reasoning in the explanation field, then determine the precise coordinates.
""",
        "detailed": """I have a table with various items captured in the image. I need to determine the 3D position of objects in the scene.

Known item coordinates (in meters from robot base):
{known_items_text}

Task: Look at the image carefully and predict the 3D coordinates of the {item_name}.

Process:
1. Identify the {item_name} in the image
2. Note its position relative to known items
3. Use the known coordinates as reference points
4. Triangulate to determine its position in 3D space
5. Consider the perspective and scale of objects

Coordinate system info:
- Origin: Robot base
- Units: Meters
- X axis: Forward direction
- Y axis: Left direction
- Z axis: Upward direction

Return your answer in this exact JSON format:
{{
    "item": "{item_name}",
    "coordinates": [x, y, z],
    "explanation": "detailed step-by-step reasoning for your coordinate prediction"
}}
""",
        "geometric": """Examine the image showing a table with various items.

GIVEN:
- Coordinate system origin at robot base
- Measurements in meters
- X axis = forward, Y axis = left, Z axis = up
- Known item coordinates:
{known_items_text}

TASK:
Determine the precise 3D coordinates of the {item_name} using geometric reasoning.

REQUIRED STEPS:
1. Estimate distances and angles between {item_name} and known reference objects
2. Calculate X, Y, Z coordinates using trigonometric principles
3. Verify consistency with overall scene geometry

RESPONSE FORMAT:
Return a JSON object:
{{
    "item": "{item_name}",
    "coordinates": [x, y, z],
    "explanation": "include your geometric calculations and reasoning"
}}
""",
        "minimal": """Known item coordinates:
{known_items_text}

Predict the coordinates of the {item_name} in meters from robot base.

Reply with JSON:
{{
    "item": "{item_name}",
    "coordinates": [x, y, z],
    "explanation": "your reasoning"
}}
""",
        "simple": """Look at the image and predict the 3D coordinates of the {item_name}.

Known item coordinates (meters from robot base):
{known_items_text}

Reasoning process:
1. Identify the {item_name} in the image
2. Note its position relative to known items
3. Use the known coordinates as reference points
4. Triangulate to determine its position in 3D space
5. Consider the perspective and scale of objects

Return ONLY a JSON object:
{{
    "item": "{item_name}",
    "explanation": "your reasoning",
    "coordinates": [x, y, z],
}}
"""
    }

def main():
    """Main function to run the experiment with hardcoded values."""
    # ===== EXPERIMENT CONFIGURATION =====
    # Set your parameters here for easy customization
    
    # Path to the image
    image_path = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/debug_images/camera_0_20250321_151203.jpg"
    
    # Path to the YAML file with known item coordinates
    yaml_path = "/home/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/llm_agent/actions_config/item_location.yaml"
    
    # Models to test (can specify multiple)
    models = ["google/gemma-3-27b-it:free",'mistralai/mistral-small-3.1-24b-instruct:free',"qwen/qwen2.5-vl-72b-instruct:free"]
    # For example, to test multiple models:
    # models = ["qwen/qwen2.5-vl-72b-instruct:free", "anthropic/claude-3-opus-20240229"]
    
    # Items to predict coordinates for
    items_to_predict = ["blender"]
    # For example, to predict multiple items:
    # items_to_predict = ["microwave", "bowl", "cup"]
    
    # Number of predictions to make per model
    num_predictions = 5
    
    # Directory to save experiment results
    output_dir = "experiment_results"
    
    # ===== PROMPT TEMPLATE SELECTION =====
    # Choose a prompt template or set to None to use standard
    # Options: "standard", "explanation_first", "detailed", "geometric", "minimal", "simple" or None
    prompt_type = "simple"
    
    # Get prompt template if specified
    prompt_template = None
    if prompt_type:
        templates = get_prompt_templates()
        if prompt_type in templates:
            prompt_template = templates[prompt_type]
            print(f"\nUsing prompt template: {prompt_type}")
        else:
            print(f"Warning: Unknown prompt type '{prompt_type}'. Using standard template.")
    else:
        print("\nUsing standard prompt template")
    
    # ===== CUSTOM PROMPT OPTIONS =====
    # Choose whether to use a custom prompt
    use_custom_prompt = False
    
    # If using custom prompt, which field order to use
    coordinates_first = True  # Set to False to put explanation first
    
    # Determine experiment name
    if use_custom_prompt:
        experiment_name = f"custom_{'coords_first' if coordinates_first else 'explanation_first'}"
    elif prompt_type:
        experiment_name = prompt_type
    else:
        experiment_name = "standard"
    
    # ===== CREATE CUSTOM PROMPT =====
    if use_custom_prompt:
        # Choose format based on coordinates_first
        if coordinates_first:
            order_instruction = "First provide the precise coordinates, then explain your reasoning."
            # Create the full prompt template for coordinates first
            prompt_template = f"""I have a table with various items and need to determine the 3D position of objects in the scene.

Known item coordinates (in meters from robot base):
{{known_items_text}}

Task: Analyze the image carefully and predict the 3D coordinates of the {{item_name}}.

Your reasoning process:
1. Identify the {{item_name}} in the image
2. Note its position relative to known items
3. Use the known coordinates as reference points
4. Triangulate to determine its position in 3D space
5. Consider the perspective and scale of objects

VERY IMPORTANT: Return your answer in this EXACT JSON format with no other text before or after:

```json
{{{{
    "item": "{{{{item_name}}}}",
    "coordinates": [x, y, z],
    "explanation": "your detailed reasoning here"
}}}}
```

Use only numbers (not strings) for the coordinates, with no units. Express coordinates in meters.
{order_instruction}
Do not include any explanatory text outside of the JSON.
"""
        else:
            order_instruction = "First provide your detailed reasoning in the explanation field, then carefully determine the precise coordinates."
            # Create the full prompt template for explanation first
            prompt_template = f"""I have a table with various items and need to determine the 3D position of objects in the scene.

Known item coordinates (in meters from robot base):
{{known_items_text}}

Task: Analyze the image carefully and predict the 3D coordinates of the {{item_name}}.

Your reasoning process:
1. Identify the {{item_name}} in the image
2. Note its position relative to known items
3. Use the known coordinates as reference points
4. Triangulate to determine its position in 3D space
5. Consider the perspective and scale of objects

VERY IMPORTANT: Return your answer in this EXACT JSON format with no other text before or after:

```json
{{{{
    "item": "{{{{item_name}}}}",
    "explanation": "your detailed reasoning here",
    "coordinates": [x, y, z]
}}}}
```

Use only numbers (not strings) for the coordinates, with no units. Express coordinates in meters.
{order_instruction}
Do not include any explanatory text outside of the JSON.
"""
        print(f"\nUsing custom prompt template with {'coordinates first' if coordinates_first else 'explanation first'}")
    
    # Create and run the experiment
    experiment = WaypointPredictionExperiment(
        image_path=image_path,
        item_location_yaml=yaml_path,
        models=models,
        num_predictions=num_predictions,
        items_to_predict=items_to_predict,
        output_dir=output_dir,
        prompt_template=prompt_template,
        experiment_name=experiment_name
    )
    
    # Run the experiment
    experiment.run_experiment()
    
    # Print summary
    experiment.print_summary()


if __name__ == "__main__":
    main()
