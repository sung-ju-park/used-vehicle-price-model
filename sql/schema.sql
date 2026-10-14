-- 중고차 매물 데이터베이스 (SQLite)
-- 원본 → 정제 → 학습용으로 단계를 나눠 저장한다. 각 단계는 앞 단계의 id를 참조한다.
--
--   raw_listings    원본 매물 그대로 (426,880건)
--   clean_listings  정제 규칙을 통과한 매물. 같은 차를 묶는 vehicle_key를 붙인다 (중복 포함)
--   vehicles        같은 차는 가장 최근 게시물 하나만 남긴 차량 단위 데이터와 데이터 분리(split)
--   training_data   학습·평가에 쓰는 뷰 (vehicles + 원본 정보)

PRAGMA foreign_keys = ON;

DROP VIEW IF EXISTS training_data;
DROP TABLE IF EXISTS vehicles;
DROP TABLE IF EXISTS clean_listings;
DROP TABLE IF EXISTS raw_listings;

CREATE TABLE raw_listings (
    id            INTEGER PRIMARY KEY,      -- Craigslist 게시물 ID (클수록 최근 게시물)
    region        TEXT,
    price         INTEGER,                  -- 판매자가 올린 가격 (달러)
    year          INTEGER,
    manufacturer  TEXT,
    model         TEXT,                     -- 자유 입력 모델명 (예: 'f150 xlt crew cab')
    condition     TEXT,
    cylinders     TEXT,
    fuel          TEXT,
    odometer      REAL,                     -- 마일
    title_status  TEXT,
    transmission  TEXT,
    vin           TEXT,
    drive         TEXT,
    size          TEXT,
    type          TEXT,
    paint_color   TEXT,
    state         TEXT
);

CREATE TABLE clean_listings (
    id           INTEGER PRIMARY KEY REFERENCES raw_listings(id),
    model_base   TEXT NOT NULL CHECK (length(model_base) > 0),   -- 정규화한 기본 모델명 (예: 'f-150')
    vehicle_key  TEXT NOT NULL                                    -- VIN, 없으면 제조사·모델·연식·주행거리(·가격)
);
CREATE INDEX idx_clean_vehicle ON clean_listings(vehicle_key);

CREATE TABLE vehicles (
    vehicle_key  TEXT PRIMARY KEY,
    listing_id   INTEGER NOT NULL UNIQUE REFERENCES clean_listings(id),   -- 남긴 게시물 (가장 최근)
    n_postings   INTEGER NOT NULL CHECK (n_postings >= 1),                 -- 같은 차가 올라온 횟수
    split        TEXT NOT NULL CHECK (split IN ('fit', 'es', 'valid', 'calib', 'test'))
);
CREATE INDEX idx_vehicles_split ON vehicles(split);

CREATE VIEW training_data AS
SELECT v.split, v.vehicle_key, v.n_postings, c.model_base, r.*
FROM vehicles v
JOIN clean_listings c ON c.id = v.listing_id
JOIN raw_listings r ON r.id = v.listing_id;
