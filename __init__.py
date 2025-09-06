"""
Server package init: ensure project root is on sys.path so routers can import models.
"""

import sys
from pathlib import Path

# Add the project root (parent of 'server') to sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
