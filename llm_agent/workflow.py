import os
import sys
import time
import argparse

from dotenv import load_dotenv
load_dotenv()  # Load .env variables

# Import the three agents
from .action_planning_agent import ActionPlanningAgent
from .action_execution_agent import ActionExecutionAgent
from .progress_monitoring_agent import ProgressMonitoringAgent

class KitchenAssistiveRobotWorkflow:
    """
    A class that implements the complete workflow for the kitchen assistive robot,
    combining action planning, execution, and progress monitoring.
    """
    
    def __init__(self, server_url=None, camera_interface=None, robot_controller=None):
        """
        Initialize the workflow with the three agents.
        
        Args:
            server_url (str, optional): URL of the server to upload images to.
            camera_interface: Optional camera interface to use for capturing frames.
            robot_controller: Optional robot controller for executing actions.
        """
        self.server_url = server_url or os.getenv("IMAGE_SERVER_URL")
        self.camera_interface = camera_interface
        self.robot_controller = robot_controller
        
        # Initialize the three agents
        self.planning_agent = ActionPlanningAgent(server_url=self.server_url)
        self.execution_agent = ActionExecutionAgent(server_url=self.server_url)
        self.monitoring_agent = ProgressMonitoringAgent(server_url=self.server_url)
        
        # Initialize workflow state
        self.user_goal = None
        self.action_plan = None
        self.execution_results = None
        self.feedback = None
    
    def run_workflow(self, user_goal):
        """
        Run the complete workflow for the given user goal.
        
        Args:
            user_goal (str): The goal specified by the user.
            
        Returns:
            dict: The results of the workflow, including action plan, execution results, and feedback.
        """
        self.user_goal = user_goal
        print(f"\n=== Starting workflow for goal: '{user_goal}' ===\n")
        
        # Step 1: Action Planning
        print("Step 1: Generating action plan...")
        action_plan_text = self.planning_agent.generate_action_plan(
            user_goal, 
            camera_interface=self.camera_interface
        )
        self.action_plan = self.planning_agent.parse_action_plan(action_plan_text)
        
        print(f"Generated action plan with {len(self.action_plan)} steps:")
        for i, step in enumerate(self.action_plan, 1):
            print(f"  {i}. {step}")
        
        # Step 2: Action Execution
        print("\nStep 2: Executing action plan...")
        self.execution_agent.set_action_plan(self.action_plan)
        self.execution_results = self.execution_agent.execute_all_steps(
            camera_interface=self.camera_interface,
            robot_controller=self.robot_controller
        )
        
        execution_status = self.execution_agent.get_execution_status()
        print(f"Executed {execution_status['completed_steps']} of {execution_status['total_steps']} steps.")
        
        # Step 3: Progress Monitoring and Feedback
        print("\nStep 3: Monitoring progress and providing feedback...")
        task_context = f"Goal: {user_goal}\nAction Plan: {', '.join(self.action_plan)}"
        self.feedback = self.monitoring_agent.monitor_progress(
            task_context=task_context,
            camera_interface=self.camera_interface
        )
        
        print(f"Feedback:\n{self.feedback}")
        
        # Return the complete workflow results
        return {
            "user_goal": self.user_goal,
            "action_plan": self.action_plan,
            "execution_results": self.execution_results,
            "execution_status": execution_status,
            "feedback": self.feedback
        }
    
    def save_workflow_results(self, output_dir="workflow_results"):
        """
        Save the workflow results to files.
        
        Args:
            output_dir (str): Directory to save the results to.
        """
        import json
        import os
        from datetime import datetime
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a results dictionary
        results = {
            "timestamp": timestamp,
            "user_goal": self.user_goal,
            "action_plan": self.action_plan,
            "execution_status": self.execution_agent.get_execution_status(),
            "feedback": self.feedback
        }
        
        # Save the results to a JSON file
        filename = os.path.join(output_dir, f"workflow_results_{timestamp}.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nWorkflow results saved to {filename}")


def main():
    """
    Main function to run the workflow from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the kitchen assistive robot workflow.")
    parser.add_argument("goal", type=str, help="The goal to accomplish (e.g., 'clean the table').")
    parser.add_argument("--server-url", type=str, help="URL of the server to upload images to.")
    parser.add_argument("--save-results", action="store_true", help="Save the workflow results to files.")
    
    args = parser.parse_args()
    
    # Create and run the workflow
    workflow = KitchenAssistiveRobotWorkflow(server_url=args.server_url)
    results = workflow.run_workflow(args.goal)
    
    # Save the results if requested
    if args.save_results:
        workflow.save_workflow_results()


if __name__ == "__main__":
    main() 