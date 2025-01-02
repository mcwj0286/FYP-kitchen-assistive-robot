import sys
import os
from libero.lifelong.models.base_policy import register_policy

# Add the workspace directory to the system path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(workspace_path)

from models.bc_baku_policy import BCBakuPolicy

# Register the policy with the registry
register_policy(BCBakuPolicy)

# Make the policy available for import
bc_baku_policy = BCBakuPolicy