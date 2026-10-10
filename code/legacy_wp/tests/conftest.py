import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))                  # code/ -> sat7
sys.path.insert(0, str(HERE.parents[1]))                  # legacy_wp/ -> wplib
