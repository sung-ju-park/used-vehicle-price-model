-- 데이터 품질 검사. 각 쿼리는 '규칙을 어긴 건수'를 돌려준다.
-- level: error 는 0이어야 통과, warn 은 건수를 기록만 한다.
-- :price_min 같은 값은 src/config.py의 정제 기준을 그대로 넘겨받는다.

-- name: price_range
-- level: error
-- desc: 학습 데이터에 가격 범위(1,000~150,000달러) 밖 매물이 있다
SELECT COUNT(*) FROM training_data WHERE price IS NULL OR price NOT BETWEEN :price_min AND :price_max;

-- name: year_range
-- level: error
-- desc: 학습 데이터에 연식 범위(1995~2022) 밖 매물이 있다
SELECT COUNT(*) FROM training_data WHERE year IS NULL OR year NOT BETWEEN :year_min AND :year_max;

-- name: odometer_rule
-- level: error
-- desc: 주행거리가 비었거나 범위 밖이거나, 2년 넘은 차인데 100마일 미만(자리표시)이다
SELECT COUNT(*) FROM training_data
WHERE odometer IS NULL OR odometer NOT BETWEEN 0 AND :odo_max
   OR (odometer < :odo_placeholder AND :ref_year - year >= 2);

-- name: required_fields
-- level: error
-- desc: 제조사·모델명·주(state)·기본 모델명 중 비어 있는 값이 있다
SELECT COUNT(*) FROM training_data
WHERE manufacturer IS NULL OR model IS NULL OR state IS NULL OR model_base IS NULL OR model_base = '';

-- name: clean_follows_rules
-- level: error
-- desc: 정제 테이블(clean_listings)에 정제 규칙을 어긴 매물이 남아 있다
SELECT COUNT(*) FROM clean_listings c JOIN raw_listings r ON r.id = c.id
WHERE r.price NOT BETWEEN :price_min AND :price_max
   OR r.year NOT BETWEEN :year_min AND :year_max
   OR r.odometer NOT BETWEEN 0 AND :odo_max
   OR (r.odometer < :odo_placeholder AND :ref_year - r.year >= 2)
   OR r.manufacturer IS NULL OR r.model IS NULL;

-- name: latest_posting_kept
-- level: error
-- desc: 같은 차의 게시물 중 가장 최근 것(ID가 가장 큰 것)을 남기지 않은 차가 있다
SELECT COUNT(*) FROM vehicles v
JOIN (SELECT vehicle_key, MAX(id) AS last_id FROM clean_listings GROUP BY vehicle_key) c USING (vehicle_key)
WHERE v.listing_id <> c.last_id;

-- name: postings_count
-- level: error
-- desc: 차량별 게시 횟수(n_postings)가 정제 테이블의 실제 게시물 수와 다르다
SELECT COUNT(*) FROM vehicles v
JOIN (SELECT vehicle_key, COUNT(*) AS n FROM clean_listings GROUP BY vehicle_key) c USING (vehicle_key)
WHERE v.n_postings <> c.n;

-- name: every_vehicle_kept
-- level: error
-- desc: 정제 테이블에는 있는데 차량 테이블에서 빠진 차가 있다
SELECT COUNT(DISTINCT vehicle_key) FROM clean_listings
WHERE vehicle_key NOT IN (SELECT vehicle_key FROM vehicles);

-- name: vin_split_leak
-- level: error
-- desc: 올바른 VIN이 같은 차가 서로 다른 데이터 묶음(학습·검증·평가 등)에 함께 들어가 있다
SELECT COUNT(*) FROM (
    SELECT UPPER(TRIM(vin)) AS v FROM training_data
    WHERE UPPER(TRIM(vin)) REGEXP '^[A-HJ-NPR-Z0-9]{17}$'
    GROUP BY v HAVING COUNT(DISTINCT split) > 1
);

-- name: invalid_vin_across_splits
-- level: warn
-- desc: 형식이 틀린 17자리 VIN(I·O·Q 포함)이 여러 묶음에 나뉘어 있다. 광고 문구를 VIN 칸에 쓴 다른 차들이거나, 0을 O로 잘못 적은 같은 차일 수 있다
SELECT COUNT(*) FROM (
    SELECT UPPER(TRIM(vin)) AS v FROM training_data
    WHERE length(TRIM(vin)) = 17 AND NOT UPPER(TRIM(vin)) REGEXP '^[A-HJ-NPR-Z0-9]{17}$'
    GROUP BY v HAVING COUNT(DISTINCT split) > 1
);

-- name: split_share
-- level: error
-- desc: 데이터 묶음 비율이 설정(학습 55·조기 종료 5·검증 10·보정 10·평가 20%)과 0.1%p 넘게 다르다
SELECT COUNT(*) FROM (
    SELECT split, COUNT(*) * 1.0 / (SELECT COUNT(*) FROM vehicles) AS share FROM vehicles GROUP BY split
) WHERE ABS(share - CASE split WHEN 'test' THEN :test_size WHEN 'valid' THEN :valid_size
                               WHEN 'calib' THEN :calib_size WHEN 'es' THEN :es_size
                               ELSE 1 - :test_size - :valid_size - :calib_size - :es_size END) > 0.001;

-- name: possible_residual_duplicate
-- level: warn
-- desc: 평가 데이터의 차 중, 제조사·모델·연식·주행거리·가격이 모두 같은 차가 다른 묶음에도 있다 (한쪽만 VIN을 적은 같은 차일 수 있음)
SELECT COALESCE(SUM(n_test), 0) FROM (
    SELECT SUM(split = 'test') AS n_test, SUM(split <> 'test') AS n_other FROM training_data
    GROUP BY manufacturer, model_base, year, odometer, price
) WHERE n_test > 0 AND n_other > 0;

-- name: placeholder_price
-- level: warn
-- desc: 111,111달러·123,456달러처럼 자리표시로 보이는 가격이 학습 데이터에 남아 있다
SELECT COUNT(*) FROM training_data
WHERE price IN (1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888,
                99999, 111111, 1234, 12345, 123456);   -- 9,999달러처럼 흔한 실제 가격은 넣지 않는다
