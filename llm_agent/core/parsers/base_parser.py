from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseParser(ABC):
    """
    Abstract base class for parsers that extract structured information from LLM responses.
    """
    def __init__(self, name: str, description: str):
        """
        Initialize the parser.
        
        Args:
            name: The name of the parser.
            description: A description of the parser's purpose.
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.
        
        Args:
            response: The raw LLM response string.
            
        Returns:
            Dict[str, Any]: The parsed response.
        """
        pass

class ActionParser(BaseParser):
    """
    Parser for extracting action and duration from LLM responses.
    """
    def __init__(
        self, 
        name: str = "action_parser",
        description: str = "Extracts action and duration from LLM responses"
    ):
        """
        Initialize the action parser.
        
        Args:
            name: The name of the parser.
            description: A description of the parser's purpose.
        """
        super().__init__(name, description)
    
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract action and duration.
        
        Args:
            response: The raw LLM response string.
            
        Returns:
            Dict[str, Any]: Dictionary containing action, duration, and analysis.
        """
        result = {
            "action": None,
            "duration": None,
            "analysis": None,
            "error": None
        }
        
        try:
            # Look for "Analysis:" and "Result:" sections
            if "Analysis:" in response and "Result:" in response:
                analysis_part = response.split("Analysis:", 1)[1].split("Result:", 1)[0].strip()
                result_part = response.split("Result:", 1)[1].strip()
                
                # Store the analysis
                result["analysis"] = analysis_part
                
                # Parse the result part to get action and duration
                parts = result_part.split(',', 1)
                
                if len(parts) != 2:
                    raise ValueError(f"Invalid format in result: '{result_part}'. Expected 'action, duration'")
                
                # Extract action (string) and duration (float/int)
                action = parts[0].strip()
                try:
                    duration = float(parts[1].strip())
                except ValueError:
                    raise ValueError(f"Invalid duration value: '{parts[1]}'")
                
                result["action"] = action
                result["duration"] = duration
                
            else:
                # Fall back to simple comma parsing if no structured format
                parts = response.split(',', 1)
                
                if len(parts) != 2:
                    raise ValueError(f"Invalid format: '{response}'. Expected 'action, duration' or 'Analysis:...Result:action, duration'")
                
                # Extract action (string) and duration (float/int)
                action = parts[0].strip()
                try:
                    duration = float(parts[1].strip())
                except ValueError:
                    raise ValueError(f"Invalid duration value: '{parts[1]}'")
                
                result["action"] = action
                result["duration"] = duration
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result

class PlanParser(BaseParser):
    """
    Parser for extracting a step-by-step plan from LLM responses.
    """
    def __init__(
        self, 
        name: str = "plan_parser",
        description: str = "Extracts a step-by-step plan from LLM responses"
    ):
        """
        Initialize the plan parser.
        
        Args:
            name: The name of the parser.
            description: A description of the parser's purpose.
        """
        super().__init__(name, description)
    
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract a step-by-step plan.
        
        Args:
            response: The raw LLM response string.
            
        Returns:
            Dict[str, Any]: Dictionary containing the parsed plan.
        """
        result = {
            "steps": [],
            "raw_plan": response,
            "error": None
        }
        
        try:
            # Split the text by newlines and filter out empty lines
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            # Extract steps (assuming they are numbered)
            steps = []
            for line in lines:
                # Look for lines that start with numbers followed by a period or parenthesis
                if line and (line[0].isdigit() or (len(line) > 1 and line[0:2].isdigit())):
                    # Extract the step content (remove the number and any delimiter)
                    step_content = line.split('.', 1)[-1].strip()
                    if not step_content and len(line.split(')', 1)) > 1:
                        step_content = line.split(')', 1)[-1].strip()
                    
                    if step_content:
                        steps.append(step_content)
            
            # If no steps were extracted, try a different approach
            if not steps and lines:
                # Just return all non-empty lines
                steps = lines
            
            result["steps"] = steps
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result

class DialogParser(BaseParser):
    """
    Parser for extracting dialog components from LLM responses.
    """
    def __init__(
        self, 
        name: str = "dialog_parser",
        description: str = "Extracts dialog components from LLM responses"
    ):
        """
        Initialize the dialog parser.
        
        Args:
            name: The name of the parser.
            description: A description of the parser's purpose.
        """
        super().__init__(name, description)
    
    def parse(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract dialog components.
        
        Args:
            response: The raw LLM response string.
            
        Returns:
            Dict[str, Any]: Dictionary containing the parsed dialog components.
        """
        result = {
            "activity": "",
            "assistance_needed": False,
            "assistance_type": "",
            "suggested_dialog": "",
            "raw_response": response,
            "error": None
        }
        
        try:
            # Split the response into lines
            lines = response.strip().split('\n')
            
            current_field = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for field markers
                if "**Activity**:" in line:
                    current_field = "activity"
                    result["activity"] = line.split("**Activity**:", 1)[1].strip()
                elif "**Needs Assistance**:" in line:
                    current_field = "assistance_needed"
                    value = line.split("**Needs Assistance**:", 1)[1].strip().lower()
                    result["assistance_needed"] = value == "yes"
                elif "**Assistance Type**:" in line:
                    current_field = "assistance_type"
                    result["assistance_type"] = line.split("**Assistance Type**:", 1)[1].strip()
                elif "**Suggested Dialog**:" in line:
                    current_field = "suggested_dialog"
                    result["suggested_dialog"] = line.split("**Suggested Dialog**:", 1)[1].strip()
                # Continue adding to the current field if it's a continuation line
                elif current_field and not line.startswith("**"):
                    result[current_field] += " " + line
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result 