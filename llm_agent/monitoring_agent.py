import re
import time
import threading
import logging
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringAgent:
    """
    Specialized agent for continuous visual monitoring and detection tasks.
    This agent analyzes camera feeds to detect specific conditions (like jar placement).
    """
    
    def __init__(self, llm_agent, camera_tool):
        """
        Initialize the monitoring agent.
        
        Args:
            llm_agent: The language model agent for image analysis
            camera_tool: Tool for capturing camera images
        """
        self.llm_agent = llm_agent
        self.camera_tool = camera_tool
        self.monitoring_active = False
        self.detection_criteria = None
        self.system_prompt = None
        self.last_analysis = None
        self.confidence_threshold = 80
        self.monitoring_thread = None
        self.monitor_id = None
        
    def start_monitoring(self, detection_criteria: str, system_prompt: str, confidence_threshold: int = 80) -> Dict[str, Any]:
        """
        Start continuous monitoring based on specified criteria.
        
        Args:
            detection_criteria: Description of what to detect (e.g., "jar on gripper")
            system_prompt: System instructions for the monitoring LLM
            confidence_threshold: Minimum confidence level (0-100) to consider detection successful
            
        Returns:
            Dict with success status and message
        """
        # Generate a unique ID for this monitoring session
        self.monitor_id = f"monitor_{int(time.time())}"
        
        self.detection_criteria = detection_criteria
        self.system_prompt = system_prompt
        self.confidence_threshold = confidence_threshold
        self.monitoring_active = True
        self.last_analysis = None
        
        logger.info(f"[{self.monitor_id}] Starting monitoring with criteria: {detection_criteria}")
        
        # Start monitoring in a separate thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        return {
            "success": True,
            "message": f"Monitoring started with criteria: {detection_criteria}"
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """
        Stop the current monitoring task.
        
        Returns:
            Dict with success status and message
        """
        if self.monitoring_active:
            logger.info(f"[{self.monitor_id}] Stopping monitoring")
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2)
            return {"success": True, "message": "Monitoring stopped"}
        return {"success": False, "message": "No active monitoring to stop"}
    
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent analysis results.
        
        Returns:
            Dict containing analysis results or None if no analysis available
        """
        return self.last_analysis
    
    def is_detection_successful(self) -> bool:
        """
        Check if the detection criteria have been met successfully.
        
        Returns:
            True if detection criteria met with confidence above threshold, False otherwise
        """
        if not self.last_analysis:
            return False
            
        return (
            self.last_analysis.get("detection_successful", False) and 
            self.last_analysis.get("confidence", 0) >= self.confidence_threshold
        )
    
    def _monitoring_loop(self) -> None:
        """
        Internal monitoring loop that runs in a separate thread.
        Continuously captures and analyzes images until monitoring is stopped.
        """
        check_interval = 2  # seconds between checks
        
        while self.monitoring_active:
            try:
                # Capture image
                capture_result = self.camera_tool("capture")
                if "Error" in capture_result:
                    logger.error(f"[{self.monitor_id}] Camera error during monitoring: {capture_result}")
                    time.sleep(check_interval)
                    continue
                
                # Get the latest image path
                image_path = self.camera_tool.get_last_image_path()
                if not image_path:
                    logger.warning(f"[{self.monitor_id}] No image path available")
                    time.sleep(check_interval)
                    continue
                
                # Create analysis prompt combining system prompt and detection criteria
                analysis_prompt = f"""
                {self.system_prompt}
                
                Detection Task: {self.detection_criteria}
                
                Analyze this image carefully and provide the following information:
                
                1. Detailed description of what you see
                2. Whether the detection criteria are met (yes/no)
                3. Your confidence level (0-100%)
                4. Any relevant observations that might help improve detection
                
                Format your response as:
                DESCRIPTION: [your detailed description]
                CRITERIA_MET: [yes/no]
                CONFIDENCE: [0-100]
                OBSERVATIONS: [any additional observations]
                """
                
                # Analyze the image
                analysis_text = self.llm_agent.analyze_captured_image(
                    self.camera_tool,
                    analysis_prompt,
                    max_tokens=500
                )
                
                # Parse the structured response
                criteria_met = "CRITERIA_MET: yes" in analysis_text
                
                # Extract confidence using regex
                confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', analysis_text)
                confidence = int(confidence_match.group(1)) if confidence_match else 0
                
                # Update last analysis
                self.last_analysis = {
                    "timestamp": time.time(),
                    "raw_analysis": analysis_text,
                    "detection_successful": criteria_met,
                    "confidence": confidence,
                    "image_path": image_path
                }
                
                # Log the monitoring update
                logger.info(f"[{self.monitor_id}] Monitoring update - Criteria met: {criteria_met}, Confidence: {confidence}%")
                
            except Exception as e:
                logger.error(f"[{self.monitor_id}] Error in monitoring loop: {str(e)}")
                
            # Sleep before next check
            time.sleep(check_interval)
            
        logger.info(f"[{self.monitor_id}] Monitoring loop ended") 