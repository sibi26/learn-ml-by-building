from gym.envs.registration import register

from catshop.envs.web_agent_site_env import WebAgentSiteEnv
from catshop.envs.web_agent_text_env import WebAgentTextEnv

register(
  id='WebAgentSiteEnv-v0',
  entry_point='catshop.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='catshop.envs:WebAgentTextEnv',
)