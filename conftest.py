import sys
from pathlib import Path

# Make the repository root importable so tests can `import models`,
# `import utils...`, `import third_party...` regardless of where pytest runs.
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
