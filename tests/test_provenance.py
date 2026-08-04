import json
from pathlib import Path

from distance_not_damage.provenance import (
    RUN_ARTIFACT_SCHEMA_VERSION,
    capture_run_provenance,
)


def test_provenance_is_json_serializable() -> None:
    provenance = capture_run_provenance(Path.cwd())

    assert provenance.schema_version == RUN_ARTIFACT_SCHEMA_VERSION
    assert provenance.python_version
    assert provenance.torch_version
    assert provenance.numpy_version
    assert provenance.git_commit is None or len(provenance.git_commit) == 40
    json.dumps(provenance.as_dict())
