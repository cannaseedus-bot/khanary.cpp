"""Validate shipped specimens and the shared negative vectors offline."""
import json
from pathlib import Path
from conformance_common import verdict


def main():
    cases = []
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
        root = Path(__file__).resolve().parents[1]
        schemas = root / "data/schema"
        registry = Registry()
        documents = {}
        for path in sorted(schemas.glob("*.schema.json")):
            schema = json.loads(path.read_text(encoding="utf-8-sig"))
            Draft202012Validator.check_schema(schema)
            documents[path.name] = schema
            resource = Resource.from_contents(schema)
            for name in (path.name, path.as_uri(), schema["$id"]):
                registry = registry.with_resource(name, resource)
        validators = {kind: Draft202012Validator(documents[name], registry=registry)
                      for kind, name in (("track", "micronaut-track.schema.json"),
                                         ("manifest", "manifest-map.schema.json"),
                                         ("advisor_submission", "advisor-submission.schema.json"))}
        for kind, folder in (("track", "tracks"), ("manifest", "manifest")):
            for path in sorted((root / "data" / folder).glob("*.json")):
                errors = list(validators[kind].iter_errors(json.loads(path.read_text(encoding="utf-8-sig"))))
                cases.append(("shipped/" + path.name, not errors))
        advisor_fixture = root / "data/conformance/advisor-submission.json"
        cases.append(("shipped/advisor-submission.json",
                      validators["advisor_submission"].is_valid(
                          json.loads(advisor_fixture.read_text(encoding="utf-8")))))
        vectors = json.loads((root / "data/conformance/contracts.json").read_text(encoding="utf-8"))
        for row in vectors:
            if row["kind"] in validators:
                cases.append((row["id"], validators[row["kind"]].is_valid(row["value"]) == row["valid"]))
        return verdict("schema", cases)
    except Exception as exc:
        return verdict("schema", cases, str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
