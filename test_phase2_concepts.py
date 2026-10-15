import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'Source Code'))

from gemini_client import _parse_concepts


def test_parse_grounded_concepts():
    result = _parse_concepts('''{"topic":"Photosynthesis","concepts":[{"name":"Definition","description":"Conversion of light energy into chemical energy","page":12},{"name":"Mechanism","description":"Light reactions produce ATP and NADPH","page":13}]}''')
    assert result['topic'] == 'Photosynthesis'
    assert len(result['concepts']) == 2
    assert result['concepts'][1]['page'] == 13
