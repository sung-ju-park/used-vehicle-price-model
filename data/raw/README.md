여기에 `vehicles.csv`를 넣습니다 (저장소에는 포함하지 않음).

- 원 출처: Kaggle "Used Cars Dataset" (Austin Reese, Craigslist 미국 중고차 매물, 2021년 수집)
- 이 프로젝트는 열 18개로 줄인 공개 버전(426,880행)을 사용했습니다. 원본에서 `id, region, price, year, manufacturer, model, condition, cylinders, fuel, odometer, title_status, transmission, VIN, drive, size, type, paint_color, state` 열만 남기면 같은 형식이 됩니다.
- 데이터 이용 조건은 원 출처에서 확인하세요.
- 이 프로젝트에서 실제로 받은 파일: GitHub `master-coder-1/practical_application_II` 저장소의 `data/vehicles.csv` (약 52MB, 426,880행)
  - https://github.com/master-coder-1/practical_application_II → `data` 폴더 → `vehicles.csv` → 다운로드 버튼(Download raw file)
  - 같은 파일인지 확인: SHA-256 `80ec856e95aa9341729d6951ffe4a47afc75960a73ebbed1c43f5ad7225c1012`
    (Windows: `certutil -hashfile data\raw\vehicles.csv SHA256`)