from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import re
import urllib.error
import urllib.parse
import urllib.request

EXCLUDED_REPOS = {"trace-rag"}
EXCLUDED_PREFIXES = ("traceops-",)

DEPLOY_FILE_HINTS = {
    "render.yaml",
    "vercel.json",
    "railway.toml",
    "railway.json",
    "netlify.toml",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "Procfile",
    ".github/workflows/ci.yml",
    ".github/workflows/deploy.yml",
    ".github/workflows/cd.yml",
    ".github/workflows/release.yml",
}

KNOWN_HEALTH_PATHS = (
    "/api/v1/health",
    "/api/health",
    "/health",
    "/docs",
)

LIVE_URL_PATTERN = re.compile(
    r"https?://[^\s)\]>'\"]+"
)

LIVE_DOMAIN_MARKERS = (
    ".onrender.com",
    ".vercel.app",
    ".railway.app",
    ".netlify.app",
    ".streamlit.app",
    ".fly.dev",
    ".herokuapp.com",
    ".run.app",
    "github.io",
)

LOCAL_ONLY_MARKERS = (
    "local only",
    "offline",
    "no hosted",
)

PENDING_MARKERS = (
    "pending verification",
    "coming soon",
)


@dataclass(slots=True)
class ReachabilityResult:
    url: str
    status: int | None
    ok: bool
    error: str | None = None


@dataclass(slots=True)
class RepoRecord:
    name: str
    default_branch: str
    html_url: str
    stack: str
    deploy_files: list[str]
    workflow_files: list[str]
    live_urls: list[str]
    health_path: str | None
    deployment_status: str
    reachability: list[ReachabilityResult]


class GitHubClient:
    def __init__(self, token: str | None = None) -> None:
        self._token = token

    def _request_json(self, url: str) -> object:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "documind-portfolio-audit"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))

    def list_repos(self, owner: str) -> list[dict[str, object]]:
        page = 1
        repos: list[dict[str, object]] = []
        while True:
            url = f"https://api.github.com/users/{owner}/repos?per_page=100&page={page}&type=owner&sort=updated"
            payload = self._request_json(url)
            if not isinstance(payload, list) or not payload:
                break
            for item in payload:
                if isinstance(item, dict):
                    repos.append(item)
            if len(payload) < 100:
                break
            page += 1
        return repos

    def get_tree_paths(self, owner: str, repo: str, default_branch: str) -> list[str]:
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        payload = self._request_json(ref_url)
        if not isinstance(payload, dict):
            return []
        tree = payload.get("tree", [])
        if not isinstance(tree, list):
            return []
        paths: list[str] = []
        for node in tree:
            if isinstance(node, dict) and node.get("type") == "blob":
                path = node.get("path")
                if isinstance(path, str):
                    paths.append(path)
        return paths

    def get_readme(self, owner: str, repo: str) -> str:
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/readme"
        try:
            payload = self._request_json(endpoint)
        except urllib.error.HTTPError:
            return ""
        if not isinstance(payload, dict):
            return ""
        content = payload.get("content")
        if not isinstance(content, str):
            return ""
        try:
            import base64

            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:
            return ""


def should_exclude_repo(repo_name: str) -> bool:
    normalized = repo_name.lower()
    if normalized in EXCLUDED_REPOS:
        return True
    return normalized.startswith(EXCLUDED_PREFIXES)


def infer_stack(paths: list[str]) -> str:
    lowered = {path.lower() for path in paths}
    has_python = any(path.endswith(".py") for path in lowered) or "requirements.txt" in lowered
    has_node = "package.json" in lowered

    if has_python and has_node:
        return "polyglot"
    if has_python:
        return "python"
    if has_node:
        return "node"
    return "unknown"


def detect_deploy_files(paths: list[str]) -> tuple[list[str], list[str]]:
    deploy_files: list[str] = []
    workflow_files: list[str] = []

    for path in sorted(paths):
        if path in DEPLOY_FILE_HINTS:
            deploy_files.append(path)
        if path.startswith(".github/workflows/") and path.endswith(".yml"):
            workflow_files.append(path)

    return deploy_files, workflow_files


def extract_live_urls(readme_text: str) -> list[str]:
    candidates = LIVE_URL_PATTERN.findall(readme_text)
    urls: list[str] = []
    for url in candidates:
        if any(marker in url for marker in LIVE_DOMAIN_MARKERS):
            urls.append(url.rstrip(".,;"))
    return sorted(dict.fromkeys(urls))


def infer_health_path(readme_text: str) -> str | None:
    for path in KNOWN_HEALTH_PATHS:
        if path in readme_text:
            return path
    return None


def classify_deployment_status(
    readme_text: str,
    deploy_files: list[str],
    live_urls: list[str],
) -> str:
    text = readme_text.lower()

    if live_urls:
        return "live_claimed"

    if any(marker in text for marker in LOCAL_ONLY_MARKERS):
        return "local_only"

    if any(marker in text for marker in PENDING_MARKERS):
        return "not_deployed"

    if deploy_files:
        return "deploy_ready"

    return "not_deployed"


def probe_url(url: str, timeout_seconds: float = 10.0) -> ReachabilityResult:
    request = urllib.request.Request(url, headers={"User-Agent": "documind-portfolio-audit"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            status = response.getcode()
            return ReachabilityResult(url=url, status=status, ok=bool(status and status < 400))
    except Exception as exc:
        return ReachabilityResult(url=url, status=None, ok=False, error=str(exc))


def build_probe_urls(live_url: str, health_path: str | None) -> list[str]:
    base = live_url.rstrip("/")
    urls = [base]
    if health_path:
        urls.append(f"{base}{health_path if health_path.startswith('/') else '/' + health_path}")
    return list(dict.fromkeys(urls))


def audit_repositories(owner: str, token: str | None = None, probe_live_urls: bool = False) -> list[RepoRecord]:
    client = GitHubClient(token=token)
    repos = client.list_repos(owner)
    records: list[RepoRecord] = []

    for repo in repos:
        name = str(repo.get("name", "")).strip()
        if not name or should_exclude_repo(name):
            continue

        default_branch = str(repo.get("default_branch", "main"))
        html_url = str(repo.get("html_url", f"https://github.com/{owner}/{name}"))

        try:
            paths = client.get_tree_paths(owner, name, default_branch)
        except urllib.error.HTTPError:
            paths = []

        deploy_files, workflow_files = detect_deploy_files(paths)
        readme_text = client.get_readme(owner, name)
        live_urls = extract_live_urls(readme_text)
        health_path = infer_health_path(readme_text)
        status = classify_deployment_status(readme_text, deploy_files, live_urls)
        stack = infer_stack(paths)

        reachability: list[ReachabilityResult] = []
        if probe_live_urls:
            for live_url in live_urls:
                for candidate_url in build_probe_urls(live_url, health_path):
                    reachability.append(probe_url(candidate_url))

        records.append(
            RepoRecord(
                name=name,
                default_branch=default_branch,
                html_url=html_url,
                stack=stack,
                deploy_files=deploy_files,
                workflow_files=workflow_files,
                live_urls=live_urls,
                health_path=health_path,
                deployment_status=status,
                reachability=reachability,
            )
        )

    return sorted(records, key=lambda item: item.name.lower())


def write_outputs(records: list[RepoRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "portfolio_registry.json"
    markdown_path = output_dir / "portfolio_dashboard.md"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "excluded_repos": sorted(EXCLUDED_REPOS),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "repos": [
            {
                "name": record.name,
                "default_branch": record.default_branch,
                "html_url": record.html_url,
                "stack": record.stack,
                "deploy_files": record.deploy_files,
                "workflow_files": record.workflow_files,
                "live_urls": record.live_urls,
                "health_path": record.health_path,
                "deployment_status": record.deployment_status,
                "reachability": [
                    {
                        "url": entry.url,
                        "status": entry.status,
                        "ok": entry.ok,
                        "error": entry.error,
                    }
                    for entry in record.reachability
                ],
            }
            for record in records
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Portfolio Deployment Dashboard",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
        "## Summary",
        "",
    ]

    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.deployment_status] = status_counts.get(record.deployment_status, 0) + 1

    for status in sorted(status_counts):
        lines.append(f"- `{status}`: {status_counts[status]}")

    lines.extend(
        [
            "",
            "## Repositories",
            "",
            "| Repo | Stack | Status | Live URL(s) | Health Path | Deploy Signals |",
            "|---|---|---|---|---|---|",
        ]
    )

    for record in records:
        live_urls = "<br>".join(record.live_urls) if record.live_urls else "-"
        deploy_signals = "<br>".join(record.deploy_files[:6]) if record.deploy_files else "-"
        lines.append(
            "| "
            f"[{record.name}]({record.html_url}) | {record.stack} | {record.deployment_status} | "
            f"{live_urls} | {record.health_path or '-'} | {deploy_signals} |"
        )

    lines.extend(
        [
            "",
            "## Execution Matrix",
            "",
            "- Python repos: `pip install -r requirements.txt && ruff check . && pytest && python -m build`",
            "- Node repos: `npm ci && npm run lint && npm run type-check && npm test && npm run build`",
            "- Polyglot repos: run both Python and Node gates per surface.",
            "- Live URL probes: root, then health path when discovered in README.",
        ]
    )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
