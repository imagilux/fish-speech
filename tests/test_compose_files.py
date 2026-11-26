"""The compose files must parse: Docker Compose rejects a key defined twice."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parent.parent


def _duplicate_keys(node):
    """Yield (key, line) for each mapping key that repeats. PyYAML itself keeps
    the last one silently, so this walks the parsed nodes instead of loading."""
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode):
                if key.value in seen:
                    yield key.value, key.start_mark.line + 1
                seen.add(key.value)
            yield from _duplicate_keys(value)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _duplicate_keys(item)


@pytest.mark.parametrize(
    "path", sorted(ROOT.glob("compose*.yml")), ids=lambda path: path.name
)
def test_compose_file_has_no_duplicate_keys(path):
    tree = yaml.compose(path.read_text(), Loader=yaml.SafeLoader)

    assert list(_duplicate_keys(tree)) == []
