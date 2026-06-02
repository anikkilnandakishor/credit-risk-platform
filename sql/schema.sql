-- Credit Risk Platform — SQLite schema
-- Primary table: customers (loaded from application_train.csv)
-- Initialize via: python -m src.data.database --load

CREATE TABLE IF NOT EXISTS customers (
    SK_ID_CURR              INTEGER PRIMARY KEY,
    TARGET                  INTEGER,
    NAME_CONTRACT_TYPE      TEXT,
    CODE_GENDER             TEXT,
    FLAG_OWN_CAR            TEXT,
    FLAG_OWN_REALTY         TEXT,
    CNT_CHILDREN            INTEGER,
    AMT_INCOME_TOTAL        REAL,
    AMT_CREDIT              REAL,
    AMT_ANNUITY             REAL,
    AMT_GOODS_PRICE         REAL,
    NAME_TYPE_SUITE         TEXT,
    NAME_INCOME_TYPE        TEXT,
    NAME_EDUCATION_TYPE     TEXT,
    NAME_FAMILY_STATUS      TEXT,
    NAME_HOUSING_TYPE       TEXT,
    REGION_POPULATION_RELATIVE REAL,
    DAYS_BIRTH              INTEGER,
    DAYS_EMPLOYED           INTEGER,
    DAYS_REGISTRATION       REAL,
    DAYS_ID_PUBLISH         INTEGER,
    OWN_CAR_AGE             REAL,
    FLAG_MOBIL              INTEGER,
    FLAG_EMP_PHONE          INTEGER,
    FLAG_WORK_PHONE         INTEGER,
    FLAG_CONT_MOBILE        INTEGER,
    FLAG_PHONE              INTEGER,
    FLAG_EMAIL              INTEGER,
    OCCUPATION_TYPE         TEXT,
    CNT_FAM_MEMBERS         REAL,
    REGION_RATING_CLIENT    INTEGER,
    REGION_RATING_CLIENT_W_CITY INTEGER,
    EXT_SOURCE_1            REAL,
    EXT_SOURCE_2            REAL,
    EXT_SOURCE_3            REAL
    -- Additional columns are created automatically when loading the full CSV
);

CREATE INDEX IF NOT EXISTS idx_customers_target ON customers(TARGET);
CREATE INDEX IF NOT EXISTS idx_customers_income ON customers(AMT_INCOME_TOTAL);
