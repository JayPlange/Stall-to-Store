-- Stall to Store schema, version 1. Prices are integer pesewas.
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    video_path    TEXT,
    status        TEXT NOT NULL CHECK (status IN ('created','ingested','processing','draft_ready','published','failed')),
    stage         TEXT,
    error         TEXT,
    source        TEXT NOT NULL CHECK (source IN ('real','fixture')),
    review_token  TEXT NOT NULL UNIQUE,
    meta_json     TEXT
);

CREATE TABLE IF NOT EXISTS products (
    id                   INTEGER PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES runs(id),
    track_id             TEXT,
    brand                TEXT,
    name                 TEXT,
    size                 TEXT,
    keyframe_path        TEXT,
    cutout_path          TEXT,
    image_path           TEXT,
    price_pesewas        INTEGER CHECK (price_pesewas IS NULL OR price_pesewas > 0),
    price_source         TEXT NOT NULL DEFAULT 'none' CHECK (price_source IN ('auto','seller','none')),
    match_status         TEXT NOT NULL DEFAULT 'no_price' CHECK (match_status IN ('auto','needs_seller','no_price')),
    match_confidence     REAL,
    match_note           TEXT,
    status               TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft','approved','hidden')),
    -- What the pipeline proposed, kept for evaluation after seller edits.
    model_brand          TEXT,
    model_name           TEXT,
    model_size           TEXT,
    model_price_pesewas  INTEGER,
    model_confidence     REAL,
    updated_at           TEXT
);
CREATE INDEX IF NOT EXISTS idx_products_run ON products(run_id);

CREATE TABLE IF NOT EXISTS price_mentions (
    id                   INTEGER PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES runs(id),
    text                 TEXT NOT NULL,
    amount_pesewas       INTEGER NOT NULL CHECK (amount_pesewas > 0),
    t_start              REAL,
    t_end                REAL,
    referent             TEXT,
    source               TEXT NOT NULL CHECK (source IN ('model','regex','fixture')),
    assigned_product_id  INTEGER REFERENCES products(id)
);

CREATE TABLE IF NOT EXISTS candidates (
    id                INTEGER PRIMARY KEY,
    mention_id        INTEGER NOT NULL REFERENCES price_mentions(id),
    product_id        INTEGER NOT NULL REFERENCES products(id),
    rank              INTEGER NOT NULL,
    score             REAL NOT NULL,
    model_rank        INTEGER,
    model_confidence  REAL
);

CREATE TABLE IF NOT EXISTS product_edits (
    id          INTEGER PRIMARY KEY,
    product_id  INTEGER NOT NULL REFERENCES products(id),
    field       TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shops (
    id               INTEGER PRIMARY KEY,
    run_id           TEXT NOT NULL UNIQUE REFERENCES runs(id),
    slug             TEXT NOT NULL UNIQUE,
    seller_name      TEXT NOT NULL,
    seller_whatsapp  TEXT NOT NULL,
    published_at     TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id             INTEGER PRIMARY KEY,
    shop_id        INTEGER NOT NULL REFERENCES shops(id),
    reference      TEXT NOT NULL UNIQUE,
    customer_wa    TEXT,
    channel        TEXT NOT NULL CHECK (channel IN ('click_to_chat','agent')),
    status         TEXT NOT NULL CHECK (status IN ('proposed','confirmed','cancelled','expired')),
    lines_json     TEXT NOT NULL,
    total_pesewas  INTEGER NOT NULL CHECK (total_pesewas > 0),
    proposal_hash  TEXT NOT NULL UNIQUE,
    created_at     TEXT NOT NULL,
    expires_at     TEXT NOT NULL,
    confirmed_at   TEXT
);

CREATE TABLE IF NOT EXISTS model_calls (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT,
    stage       TEXT NOT NULL,
    model       TEXT NOT NULL,
    provider    TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    http_status INTEGER,
    error_type  TEXT,
    latency_ms  INTEGER,
    units_json  TEXT,
    cost_usd    REAL,
    raw_path    TEXT,
    created_at  TEXT NOT NULL
);
