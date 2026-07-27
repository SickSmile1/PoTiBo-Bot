import sys
from pathlib import Path

# Add the app directory to sys.path so imports like `from bot import DocBot` work.
sys.path.insert(0, str(Path(__file__).parent.parent))
