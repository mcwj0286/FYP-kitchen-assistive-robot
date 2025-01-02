from libero.lifelong.models.base_policy import get_policy_class, get_policy_list
from libero.lifelong.models.bc_rnn_policy import BCRNNPolicy 
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy
from libero.lifelong.models.bc_vilt_policy import BCViLTPolicy
from libero.lifelong.models.baku_policy import bc_baku_policy  # Make sure this is imported

# The registration happens when the module is imported
