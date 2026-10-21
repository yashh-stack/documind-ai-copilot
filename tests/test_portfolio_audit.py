from app.portfolio_audit import (
    build_probe_urls,
    classify_deployment_status,
    detect_deploy_files,
    extract_live_urls,
    infer_stack,
    should_exclude_repo,
)


def test_traceops_repos_are_excluded() -> None:
    assert should_exclude_repo("trace-rag") is True
    assert should_exclude_repo("traceops-sre-engine") is True
    assert should_exclude_repo("documind-ai-copilot") is False


def test_detect_deploy_files_and_workflows() -> None:
    deploy_files, workflow_files = detect_deploy_files(
        [
            "README.md",
            "render.yaml",
            "frontend/vercel.json",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
        ]
    )

    assert deploy_files == [".github/workflows/ci.yml", ".github/workflows/release.yml", "render.yaml"]
    assert workflow_files == [".github/workflows/ci.yml", ".github/workflows/release.yml"]


def test_extract_live_urls_filters_known_hosts() -> None:
    readme = """
    Live URL: https://customer-inquiry-classifier.vercel.app/
    docs: https://example.com/docs
    Health: https://documind-ai-copilot.onrender.com/api/v1/health
    """
    urls = extract_live_urls(readme)

    assert urls == [
        "https://customer-inquiry-classifier.vercel.app/",
        "https://documind-ai-copilot.onrender.com/api/v1/health",
    ]


def test_classification_precedence() -> None:
    assert classify_deployment_status("", ["render.yaml"], ["https://a.vercel.app"]) == "live_claimed"
    assert classify_deployment_status("offline local only cli", [], []) == "local_only"
    assert classify_deployment_status("public URL pending verification", ["render.yaml"], []) == "not_deployed"
    assert classify_deployment_status("", ["render.yaml"], []) == "deploy_ready"
    assert classify_deployment_status("", [], []) == "not_deployed"


def test_stack_inference_and_probe_urls() -> None:
    assert infer_stack(["requirements.txt", "app/main.py"]) == "python"
    assert infer_stack(["package.json", "src/index.ts"]) == "node"
    assert infer_stack(["requirements.txt", "package.json"]) == "polyglot"

    assert build_probe_urls("https://x.vercel.app/", "/api/health") == [
        "https://x.vercel.app",
        "https://x.vercel.app/api/health",
    ]
