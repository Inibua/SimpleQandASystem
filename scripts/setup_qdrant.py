import subprocess
import time
import urllib.request


QDRANT_CONTAINER_NAME = "qdrant"
QDRANT_IMAGE = "qdrant/qdrant:latest"
QDRANT_PORT = 6333
QDRANT_HEALTH_URL = f"http://localhost:{QDRANT_PORT}/healthz"


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def container_exists() -> bool:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={QDRANT_CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return QDRANT_CONTAINER_NAME in result.stdout.splitlines()


def start_qdrant() -> None:
    if container_exists():
        run_command(["docker", "start", QDRANT_CONTAINER_NAME])
    else:
        run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                QDRANT_CONTAINER_NAME,
                "-p",
                f"{QDRANT_PORT}:6333",
                QDRANT_IMAGE,
            ]
        )


def verify_qdrant() -> None:
    time.sleep(2)  # allow container to start
    try:
        with urllib.request.urlopen(QDRANT_HEALTH_URL, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Qdrant health check failed")
    except Exception as exc:
        raise RuntimeError("Unable to reach Qdrant health endpoint") from exc


def main() -> None:
    start_qdrant()
    verify_qdrant()


if __name__ == "__main__":
    main()
