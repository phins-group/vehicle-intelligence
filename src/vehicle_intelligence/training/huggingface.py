"""Private Hugging Face Hub registry and pay-as-you-go Job adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from vehicle_intelligence.exceptions import ModelRegistryError
from vehicle_intelligence.training.artifacts import verify_model_package
from vehicle_intelligence.training.dataset import verify_detector_dataset
from vehicle_intelligence.training.domain import HubJobResult, HubUploadResult


class HuggingFacePrivateRegistry:
    """Upload only verified immutable folders into explicitly private repos."""

    def __init__(self, *, token: str | None = None, api: Any | None = None) -> None:
        self._token = token
        self._api = api or _hf_api(token)

    def upload_dataset(
        self,
        directory: Path,
        repo_id: str,
        *,
        revision: str = "main",
        allow_restricted_private: bool = False,
        replace_remote: bool = False,
    ) -> HubUploadResult:
        folder = directory.expanduser().resolve()
        manifest, _ = verify_detector_dataset(folder)
        _validate_dataset_hub_metadata(
            folder,
            manifest,
            allow_restricted_private=allow_restricted_private,
        )
        return self._upload(
            folder,
            repo_id,
            "dataset",
            revision,
            replace_remote=replace_remote,
        )

    def upload_model(
        self,
        directory: Path,
        repo_id: str,
        *,
        revision: str = "main",
    ) -> HubUploadResult:
        folder = directory.expanduser().resolve()
        verify_model_package(folder)
        return self._upload(folder, repo_id, "model", revision)

    def _upload(
        self,
        folder: Path,
        repo_id: str,
        repo_type: str,
        revision: str,
        *,
        replace_remote: bool = False,
    ) -> HubUploadResult:
        _validate_repo_id(repo_id)
        if not revision.strip() or len(revision) > 128:
            raise ModelRegistryError("Hugging Face revision is invalid")
        try:
            self._api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                private=True,
                exist_ok=True,
                token=self._token,
            )
            info = self._api.repo_info(
                repo_id=repo_id,
                repo_type=repo_type,
                token=self._token,
            )
            if getattr(info, "private", None) is not True:
                raise ModelRegistryError("refusing to upload into a non-private Hugging Face repo")
            upload_options: dict[str, Any] = {
                "folder_path": str(folder),
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "commit_message": f"Upload verified {folder.name}",
                "token": self._token,
            }
            if replace_remote:
                upload_options["delete_patterns"] = "*"
            commit = self._api.upload_folder(
                **upload_options,
            )
        except ModelRegistryError:
            raise
        except Exception as exc:
            raise ModelRegistryError(f"private Hugging Face {repo_type} upload failed") from exc
        return HubUploadResult(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=str(getattr(commit, "oid", revision) or revision),
            url=str(getattr(commit, "commit_url", "") or "") or None,
        )


class HuggingFaceJobRunner:
    """Submit a custom training container with a read-only private dataset mount."""

    def __init__(
        self,
        *,
        run_job: Callable[..., Any] | None = None,
        volume_factory: Callable[..., Any] | None = None,
    ) -> None:
        if run_job is None or volume_factory is None:
            imported_run_job, imported_volume = _job_dependencies()
            run_job = run_job or imported_run_job
            volume_factory = volume_factory or imported_volume
        self._run_job = run_job
        self._volume = volume_factory

    def submit(
        self,
        *,
        image: str,
        command: Sequence[str],
        flavor: str,
        dataset_repo: str,
        dataset_revision: str | None = None,
        output_bucket: str,
        namespace: str | None = None,
        timeout_seconds: int = 86_400,
        environment: Mapping[str, str] | None = None,
        secrets: Mapping[str, str] | None = None,
        name: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> HubJobResult:
        _validate_repo_id(dataset_repo)
        _validate_repo_id(output_bucket)
        if (
            not image.strip()
            or not flavor.strip()
            or not command
            or any(not item for item in command)
        ):
            raise ModelRegistryError("Hugging Face Job image/flavor/command is invalid")
        if not 60 <= timeout_seconds <= 2_592_000:
            raise ModelRegistryError("Hugging Face Job timeout is outside safe limits")
        dataset_volume = self._volume(
            type="dataset",
            source=dataset_repo,
            mount_path="/data",
            revision=dataset_revision,
            read_only=True,
        )
        output_volume = self._volume(
            type="bucket",
            source=output_bucket,
            mount_path="/output",
            read_only=False,
        )
        try:
            job = self._run_job(
                image=image,
                command=list(command),
                flavor=flavor,
                namespace=namespace,
                timeout=timeout_seconds,
                env=dict(environment or {}),
                secrets=dict(secrets or {}),
                volumes=[dataset_volume, output_volume],
                name=name,
                labels=dict(labels or {}),
            )
        except Exception as exc:
            raise ModelRegistryError("Hugging Face training Job submission failed") from exc
        status = getattr(job, "status", None)
        stage = getattr(status, "stage", None) if status is not None else None
        return HubJobResult(
            job_id=str(getattr(job, "id", "")),
            url=str(getattr(job, "url", "") or "") or None,
            status=str(stage) if stage is not None else None,
        )


def _hf_api(token: str | None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ModelRegistryError(
            "Hugging Face integration requires `pip install -e '.[training]'`"
        ) from exc
    return HfApi(token=token)


def _job_dependencies() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from huggingface_hub import Volume, run_job
    except ImportError as exc:
        raise ModelRegistryError(
            "Hugging Face Jobs require `pip install -e '.[training]'`"
        ) from exc
    return run_job, Volume


def _validate_repo_id(value: str) -> None:
    stripped = value.strip().strip("/")
    parts = stripped.split("/")
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        raise ModelRegistryError("Hugging Face repository id must be namespace/name")


def _validate_dataset_hub_metadata(
    folder: Path,
    manifest: Mapping[str, Any],
    *,
    allow_restricted_private: bool = False,
) -> None:
    if manifest.get("distributionEligible") is False and not allow_restricted_private:
        raise ModelRegistryError(
            "detector dataset is not distribution-eligible; resolve source licensing first"
        )
    if manifest.get("distributionEligible") is False:
        source = manifest.get("source")
        if (
            not isinstance(source, Mapping)
            or manifest.get("releaseEligible") is not True
            or source.get("type") != "FIRST_PARTY_SOURCE"
            or source.get("rightsAssertion")
            not in {
                "USER_CONFIRMED_FIRST_PARTY_COLLECTION",
                "USER_CONFIRMED_FIRST_PARTY_VIDEO_COLLECTION",
            }
            or source.get("licenseReviewStatus")
            != "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED"
        ):
            raise ModelRegistryError(
                "restricted private Hub sync requires a release-eligible first-party dataset"
            )
    recorded = {
        entry.get("path")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    }
    required = {"README.md", "ATTRIBUTION.csv"}
    if manifest.get("acceptanceEligible") is False:
        required.add("BOOTSTRAP_ONLY.md")
    if not required.issubset(recorded):
        raise ModelRegistryError(
            "detector dataset lacks verified Hub license/provenance metadata; rebuild it"
        )
    try:
        card = (folder / "README.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelRegistryError("detector dataset card is unreadable") from exc
    if not card.startswith("---\nlicense: other\n"):
        raise ModelRegistryError("detector dataset card must declare license: other")
