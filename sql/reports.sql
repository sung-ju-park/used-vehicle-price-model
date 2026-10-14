-- 리포트에 쓴 주요 숫자를 SQL로 다시 계산한다. build_db.py가 pandas 결과(results.json)와 맞춰 본다.

-- name: funnel_price
-- desc: 가격 범위 규칙을 통과한 매물 수
SELECT COUNT(*) FROM raw_listings WHERE price BETWEEN :price_min AND :price_max;

-- name: funnel_year
-- desc: + 연식 범위
SELECT COUNT(*) FROM raw_listings
WHERE price BETWEEN :price_min AND :price_max AND year BETWEEN :year_min AND :year_max;

-- name: funnel_odometer
-- desc: + 주행거리 규칙
SELECT COUNT(*) FROM raw_listings
WHERE price BETWEEN :price_min AND :price_max AND year BETWEEN :year_min AND :year_max
  AND odometer BETWEEN 0 AND :odo_max AND NOT (odometer < :odo_placeholder AND :ref_year - year >= 2);

-- name: funnel_model
-- desc: + 제조사·모델명 있음
SELECT COUNT(*) FROM raw_listings
WHERE price BETWEEN :price_min AND :price_max AND year BETWEEN :year_min AND :year_max
  AND odometer BETWEEN 0 AND :odo_max AND NOT (odometer < :odo_placeholder AND :ref_year - year >= 2)
  AND manufacturer IS NOT NULL AND model IS NOT NULL;

-- name: funnel_clean
-- desc: + 모델명 해석 가능 (정제 테이블 행 수)
SELECT COUNT(*) FROM clean_listings;

-- name: funnel_vehicles
-- desc: 같은 차는 최신 게시물 하나만 (차량 수)
SELECT COUNT(*) FROM vehicles;

-- name: vin_rows
-- desc: VIN이 적힌 원본 매물 수
SELECT COUNT(*) FROM raw_listings WHERE vin IS NOT NULL;

-- name: vin_unique
-- desc: 서로 다른 VIN 수
SELECT COUNT(DISTINCT UPPER(vin)) FROM raw_listings WHERE vin IS NOT NULL;

-- name: reposted_vins
-- desc: 두 번 이상 올라온 VIN 수
SELECT COUNT(*) FROM (SELECT UPPER(vin) FROM raw_listings WHERE vin IS NOT NULL GROUP BY UPPER(vin) HAVING COUNT(*) > 1);

-- name: reposted_rows
-- desc: 반복 게시된 VIN의 게시물 수 (원본의 44%)
SELECT SUM(n) FROM (SELECT COUNT(*) AS n FROM raw_listings WHERE vin IS NOT NULL GROUP BY UPPER(vin) HAVING COUNT(*) > 1);

-- name: max_reposts
-- desc: 한 VIN이 가장 많이 올라온 횟수
SELECT MAX(n) FROM (SELECT COUNT(*) AS n FROM raw_listings WHERE vin IS NOT NULL GROUP BY UPPER(vin));

-- name: same_price_share
-- desc: 반복 게시된 차 중 모든 게시물의 가격이 같은 비율
SELECT AVG(np = 1) FROM (
    SELECT COUNT(DISTINCT price) AS np FROM raw_listings WHERE vin IS NOT NULL
    GROUP BY UPPER(vin) HAVING COUNT(*) > 1
);

-- name: multi_region_share
-- desc: 반복 게시된 차 중 여러 지역에 올라온 비율
SELECT AVG(nr > 1) FROM (
    SELECT COUNT(DISTINCT region) AS nr FROM raw_listings WHERE vin IS NOT NULL
    GROUP BY UPPER(vin) HAVING COUNT(*) > 1
);

-- name: truck_share
-- desc: 차량 중 트럭·픽업 비율
SELECT AVG(COALESCE(type IN ('truck', 'pickup'), 0)) FROM training_data;

-- name: test_rows
-- desc: 평가 데이터 차량 수
SELECT COUNT(*) FROM vehicles WHERE split = 'test';
