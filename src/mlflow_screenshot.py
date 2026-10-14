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
            page = b.new_page(viewport={"width": 1700, "height": 900}, device_scale_factor=1.5)
            page.goto(f"http://127.0.0.1:{PORT}/#/experiments/{exp.experiment_id}/runs", wait_until="load")
            time.sleep(10)
            page.mouse.click(1657, 35)   # 오른쪽 어시스턴트 패널 닫기
            page.mouse.click(173, 24)    # 왼쪽 메뉴 접기
            time.sleep(1)
            # 검증 지표(valid_*), 최종 평가 지표(test_*), 평가 방식 점검 지표(mdape, within_20)만 보이게 한다
            page.get_by_role("button", name="Columns", exact=True).click()
            time.sleep(2)
            dlg = page.locator("[role=dialog]").last
            for t in ["Dataset", "Source", "valid_mdape", "valid_within_20", "valid_mae_usd", "test_mdape",
                      "test_within_20", "mdape", "within_20"]:
                dlg.get_by_text(t, exact=True).first.click()
                time.sleep(0.4)
            page.keyboard.press("Escape")
            page.mouse.move(5, 880)
            time.sleep(2)
            FIG.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(FIG / "mlflow.png"), clip={"x": 60, "y": 18, "width": 1402, "height": 578})
            b.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
