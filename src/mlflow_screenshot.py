"""MLflow 실험 화면을 캡처해 리포트에 넣는다 (선택 사항).
실행: python src/mlflow_screenshot.py   (playwright와 Chromium 필요)
"""
import subprocess
import sys
import time
import urllib.request

import mlflow

from config import EXPERIMENT, FIG, MLFLOW_ARTIFACTS, MLFLOW_URI

PORT = 5055


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    proc = subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--backend-store-uri", MLFLOW_URI,
                             "--default-artifact-root", MLFLOW_ARTIFACTS, "--port", str(PORT), "--workers", "1"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=1.5)
            page.goto(f"http://127.0.0.1:{PORT}/#/experiments/{exp.experiment_id}", wait_until="networkidle")
            time.sleep(4)
            FIG.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(FIG / "mlflow.png"))
            b.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
