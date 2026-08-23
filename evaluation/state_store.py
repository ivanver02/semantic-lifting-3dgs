# State store for sequential evaluations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


STATES = {"pending", "running", "done", "failed"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def unit_id(experiment, dataset, scene, variant):

    # Serialize the unit identity before hashing
    identity = {
        "experiment": experiment,
        "dataset": dataset,
        "scene": scene,
        "variant": variant,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class StateStore:
    """ Store experiment state in an atomic JSON file """

    def __init__(self, path, read_only=False):

        # Load the current state store
        self.path = Path(path)
        self.read_only = read_only

        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self.rows = self._read()
        if not self.read_only:
            self._recover_running()

    def _read(self):

        # Read an existing state store or start empty
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"invalid experiment state store: {self.path}") from error
        if not isinstance(data, dict):
            raise RuntimeError(f"invalid experiment state store shape: {self.path}")

    # Return the parsed state store
        return data

    def _write(self):

        # Persist the state store through a temporary file
        if self.read_only:
            raise RuntimeError("read-only state store cannot be modified")
        fd, name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.rows, handle, indent=2, sort_keys=True)

    # Flush the state store file
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None

    # Sync the state store directory
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    # Persist recovered failures
    def _recover_running(self):
        # Mark interrupted units as failed
        changed = False
        for row in self.rows.values():
            if row.get("state") == "running":
                row["state"] = "failed"
                row["finished_at"] = utc_now()
                row["error"] = "interrupted before state store recovery"
                changed = True
        if changed:

    # Store the running state
            self._write()

    def state(self, identifier):
        # Return the current unit state
        return self.rows.get(identifier, {}).get("state")

    def is_done(self, identifier, fingerprint=None):
        # Check completion and optional parameter identity
        row = self.rows.get(identifier, {})
        return (
            row.get("state") == "done" and
            (fingerprint is None or row.get("parameters_fingerprint") == fingerprint)
        )

    def is_source_done(self, identifier, source, fingerprint=None):
        # Check completion for one source
        row = self.rows.get(identifier, {})
        return (
            source in row.get("completed_sources", []) and
            (fingerprint is None or row.get("parameters_fingerprint") == fingerprint)
        )

    def begin(self, identifier, experiment, dataset, scene, variant, run_id,
              attempt=None, parameters_fingerprint=None, sources=None):
        # Create a running unit record
        previous = self.rows.get(identifier, {})
        attempt = int(attempt if attempt is not None else previous.get("attempt", 0)) + 1
        self.rows[identifier] = {
            "unit_id": identifier,
            "experiment": experiment,
            "dataset": dataset,
            "scene": scene,
            "variant": variant,

    # Store source completion fields
            "state": "running",
            "run_id": run_id,
            "started_at": utc_now(),
            "finished_at": None,
            "exit_code": None,
            "error": None,
            "attempt": attempt,
            "parameters_fingerprint": parameters_fingerprint,

    # Set the final unit state
            "sources": list(sources or []),
            "completed_sources": list(previous.get("completed_sources", [])),
        }
        self._write()

    def finish(self, identifier, exit_code=0, error=None, source=None):
        # Update source and unit completion state
        row = self.rows.setdefault(identifier, {"unit_id": identifier})
        if source is not None and exit_code == 0 and error is None:
            completed = set(row.get("completed_sources", []))
            completed.add(source)
            row["completed_sources"] = sorted(completed)
        all_sources_done = set(row.get("sources", [])) <= set(
            row.get("completed_sources", [])
        )

    # Store pending timestamps
        row["state"] = (
            "done" if exit_code == 0 and error is None and all_sources_done
            else "failed" if error is not None or exit_code != 0 else "running"
        )
        row["finished_at"] = utc_now()
        row["exit_code"] = exit_code
        row["error"] = error
        self._write()

def read_state_store(path):
    """ Read experiment state without creating or modifying the state store """
    return StateStore(path, read_only=True)