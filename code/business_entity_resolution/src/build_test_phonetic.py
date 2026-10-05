"""Build v4 supplemental test retrieval after the v3 target index exists."""
from pathlib import Path
import phonetic_index


if __name__ == '__main__':
    index = Path('artifacts/test_index')
    phonetic_index.build(index)
    phonetic_index.build_native(index)
