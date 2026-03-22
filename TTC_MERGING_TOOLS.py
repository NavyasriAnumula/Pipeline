
import psycopg2
from psycopg2.extras import Json, execute_values
from datetime import datetime
import json
from fuzzywuzzy import fuzz
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
# ============================================================================
# CONFIGURATION
# ============================================================================
from config import (
    DB_CONFIG,
    OPENAI_API_KEY_GPT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SCHEMA,
    FUZZY_SIMILARITY_THRESHOLD,
    LLM_BATCH_SIZE,
    DEFAULT_MIN_TIMESTAMP,
    CATEGORY_CONFIG,
    PIPELINE_TABLES
)

labeled_table = PIPELINE_TABLES["LABELED"]
RUN_LLM_CLASSIFICATION = False
# ============================================================================
# SET CATEGORY HERE - CHANGE THIS TO RUN DIFFERENT PIPELINES
# ============================================================================
CATEGORY = "TOOLS"  # Options: "TOOLS", "TECHNOLOGIES", "COMPLIANCES"

# Get configuration for selected category
CONFIG = CATEGORY_CONFIG[CATEGORY]
OPENAI_API_KEY = OPENAI_API_KEY_GPT


# ============================================================================
# DATABASE CONNECTION
# ============================================================================
def connect_to_db():
    """Establish database connection"""
    try:
        conn = psycopg2.connect(
            **DB_CONFIG,
            options=f"-c search_path={SCHEMA},PROCESSING"
        )
        cur = conn.cursor()
        print("✅ Connected to PostgreSQL")
        return conn, cur
    except psycopg2.Error as e:
        print(f"❌ DB Connection Failed: {e}")
        sys.exit(1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_max_timestamp(cur):
    """Get the maximum timestamp from merged data table"""
    try:
        query = f"""
            SELECT COALESCE(MAX("INSERTED_TIMESTAMP"), %s::TIMESTAMP) 
            FROM "{SCHEMA}"."{CONFIG['merged_table']}"
        """
        cur.execute(query, (DEFAULT_MIN_TIMESTAMP,))
        max_ts = cur.fetchone()[0]
        print(f"📅 Max timestamp from merged data: {max_ts}")
        return max_ts
    except Exception as e:
        print(f"⚠️  Error fetching max timestamp, using default: {e}")
        return DEFAULT_MIN_TIMESTAMP


def calculate_similarity(str1, str2):
    """Calculate similarity between two strings"""
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    return fuzz.token_sort_ratio(s1, s2)


def extract_distinct_domains(items_data):
    """Extract distinct domain values from JSON array"""
    domains = set()
    for item in items_data:
        domain = (item.get('domain') or "").strip()
        if domain:
            domains.add(domain)
    return ', '.join(sorted(domains))


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================
def fetch_existing_merged_items(cur):
    """Fetch all existing items from merged data table"""
    print(f"\n🔍 Fetching existing merged {CATEGORY.lower()} from {CONFIG['merged_table']}...")

    query = f"""
    SELECT 
        "{CONFIG['item_column']}",
        "DEFINITION",
        "SOURCE",
        "INSERTED_TIMESTAMP",
        "{CONFIG['data_json_column']}",
        "JD_DOMAIN"
    FROM "{SCHEMA}"."{CONFIG['merged_table']}"
    ORDER BY "{CONFIG['item_column']}"
    """

    cur.execute(query)
    rows = cur.fetchall()

    existing_items = []
    for row in rows:
        items_data = row[4] if row[4] else []

        # Parse JSON if it's a string
        if isinstance(items_data, str):
            try:
                items_data = json.loads(items_data)
            except:
                items_data = []

        existing_items.append({
            'item': row[0],
            'definition': row[1] or '',
            'source': row[2] or '',
            'inserted_timestamp': row[3],
            'items_data': items_data,
            'jd_domain': row[5] or ''
        })

    print(f"   ✅ Found {len(existing_items)} existing merged {CATEGORY.lower()}")
    return existing_items


def fetch_new_items_from_standardization(cur, max_timestamp):
    """Fetch new items from standardization result table — per domain in parallel"""
    print(f"\n🔧 Fetching new {CATEGORY.lower()} from {CONFIG['standardization_table']}...")
    overall_start = time.time()

    # ── STEP A: Get distinct domains first ───────────────────────────────────
    print(f"   🔍 Fetching distinct domains from {CONFIG['standardization_table']}...")
    domain_start = time.time()
    domain_query = f"""
        SELECT DISTINCT "DOMAIN"
        FROM "PROCESSING"."{CONFIG['standardization_table']}"
        WHERE "STATUS" = 'TO BE REVIEWED'
    """
    cur.execute(domain_query)
    domains = [row[0] for row in cur.fetchall()]
    domain_elapsed = time.time() - domain_start
    print(f"   ✅ Found {len(domains)} distinct domains in {domain_elapsed:.2f}s: {domains}")

    if not domains:
        print(f"   ⚠️  No domains found, skipping fetch")
        return []

    # ── STEP B: Fetch delta per domain using threads ──────────────────────────
    print(f"\n   ⚡ Fetching delta records for each domain IN PARALLEL...")

    def fetch_for_domain(domain):
        """Opens its own DB connection and fetches delta for one domain"""
        domain_conn, domain_cur = connect_to_db()
        d_start = time.time()
        try:
            query = f"""
            SELECT DISTINCT ON (LOWER(s."{CONFIG['extracted_column']}"))
                s."{CONFIG['extracted_column']}" AS item,
                s."EXTRACTED_DEFINTION" AS definition,
                s."INSERTED_TIMESTAMP" AS inserted_timestamp,
                s."JD_ID" AS jd_id,
                s."DOMAIN" AS domain,
                h."SOURCE_PHRASE" AS source_phrase,
                h."REASON" AS existing_reason,
                h."CONFIDENCE_SCORE" AS confidence_score
            FROM "PROCESSING"."{CONFIG['standardization_table']}" s
            LEFT JOIN "{SCHEMA}"."{CONFIG['high_confidence_table']}" h
                ON h."JD_ID" = s."JD_ID"
               AND LOWER(TRIM(h."{CONFIG['item_column']}")) =
                   LOWER(TRIM(s."{CONFIG['extracted_column']}"))
            WHERE s."STATUS" = 'TO BE REVIEWED'
              AND s."DOMAIN" = %s
              AND s."{CONFIG['extracted_column']}" IS NOT NULL
              AND s."{CONFIG['extracted_column']}" <> ''
              AND NOT EXISTS (
                    SELECT 1
                    FROM "PROCESSING"."{CONFIG['masterdata_table']}" m
                    WHERE LOWER(m."{CONFIG['item_column']}") =
                          LOWER(s."{CONFIG['extracted_column']}")
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM "{SCHEMA}"."{CONFIG['merged_table']}" mt
                  WHERE LOWER(mt."{CONFIG['item_column']}") =
                        LOWER(s."{CONFIG['extracted_column']}")
                     OR EXISTS (
                         SELECT 1
                         FROM json_array_elements(mt."{CONFIG['data_json_column']}"::json) jd
                         WHERE LOWER(jd->>'{CONFIG['json_item_key']}') =
                               LOWER(s."{CONFIG['extracted_column']}")
                     )
              )
            ORDER BY
                LOWER(s."{CONFIG['extracted_column']}"),
                s."INSERTED_TIMESTAMP" DESC
            """
            domain_cur.execute(query, (domain,))
            rows = domain_cur.fetchall()
            d_elapsed = time.time() - d_start
            print(f"      ✅ Domain '{domain}': {len(rows)} records fetched in {d_elapsed:.2f}s")

            items = []
            for row in rows:
                items.append({
                    'item': row[0],
                    'definition': row[1] or '',
                    'source': CONFIG['standardization_table'],
                    'inserted_timestamp': row[2],
                    'jd_id': row[3],
                    'domain': row[4],
                    'source_phrase': row[5] or '',
                    'existing_reason': row[6] or '',
                    'confidence_score': row[7] or ''
                })
            return domain, items

        except Exception as e:
            d_elapsed = time.time() - d_start
            print(f"      ❌ Domain '{domain}' failed after {d_elapsed:.2f}s: {str(e)[:100]}")
            return domain, []
        finally:
            domain_cur.close()
            domain_conn.close()

    # Fire all domain fetches in parallel
    all_items = []
    domain_results = {}
    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        futures = {executor.submit(fetch_for_domain, domain): domain for domain in domains}
        for future in as_completed(futures):
            domain, items = future.result()
            domain_results[domain] = items
            all_items.extend(items)

    # Deduplicate across domains by item name (case-insensitive), keep first seen
    print(f"\n   🔄 Deduplicating across domains...")
    dedup_start = time.time()
    seen = {}
    deduped = []
    for item in all_items:
        key = item['item'].lower().strip()
        if key not in seen:
            seen[key] = True
            deduped.append(item)
    dedup_elapsed = time.time() - dedup_start

    overall_elapsed = time.time() - overall_start

    print(f"\n   📊 Domain fetch summary:")
    for domain, items in domain_results.items():
        print(f"      {domain}: {len(items)} records")
    print(f"   🔄 Deduplication: {len(all_items)} → {len(deduped)} records in {dedup_elapsed:.3f}s")
    print(f"   ⏱️  Total fetch time: {overall_elapsed:.2f}s")
    print(f"   ✅ Final: {len(deduped)} new {CATEGORY.lower()} from standardization table")

    return deduped

def fetch_new_items_from_masterdata(cur, max_timestamp):
    """Fetch new items from masterdata table"""
    print(f"\n🔧 Fetching new {CATEGORY.lower()} from {CONFIG['masterdata_table']}...")

    query = f"""
    SELECT DISTINCT ON (LOWER("{CONFIG['item_column']}"))
        "{CONFIG['item_column']}" AS item,
        "DESCRIPTION" AS definition,
        "INSERTED_TIMESTAMP" AS inserted_timestamp
    FROM "PROCESSING"."{CONFIG['masterdata_table']}"
    WHERE "INSERTED_TIMESTAMP" > %s
    ORDER BY LOWER("{CONFIG['item_column']}"), "INSERTED_TIMESTAMP" DESC
    LIMIT 0 
    """

    cur.execute(query, (max_timestamp,))
    rows = cur.fetchall()

    items = []
    for row in rows:
        items.append({
            "item": row[0],
            "definition": row[1] or "",
            "source": CONFIG['masterdata_table'],
            "inserted_timestamp": row[2],
            "existing_reason": "",
            "source_phrase": "",
            "jd_id": "",
            "domain": ""
        })

    print(f"   ✅ Found {len(items)} new {CATEGORY.lower()} from {CONFIG['masterdata_table']}")
    return items


def fetch_unclassified_items_from_merged_data(cur):
    """Fetch all items from merged table that are not yet classified"""
    print(f"\n🔍 Fetching unclassified {CATEGORY.lower()} from {CONFIG['merged_table']}...")

    query = f"""
        SELECT 
            "{CONFIG['item_column']}",
            "DEFINITION",
            "SOURCE",
            "INSERTED_TIMESTAMP",
            "{CONFIG['data_json_column']}",
            "JD_DOMAIN"
        FROM "{SCHEMA}"."{CONFIG['merged_table']}"
        WHERE "{CONFIG['item_column']}" NOT IN (
            SELECT "EXTRACTED_NAME" 
            FROM "{SCHEMA}"."{PIPELINE_TABLES['LLM_CLASSIFICATION']}"
        )
        ORDER BY "INSERTED_TIMESTAMP" DESC 
    """

    cur.execute(query)
    rows = cur.fetchall()

    items_to_classify = []
    for row in rows:
        items_data = row[4] if row[4] else []

        if isinstance(items_data, str):
            try:
                items_data = json.loads(items_data)
            except:
                items_data = []

        jd_id = ""
        source_phrase = ""
        existing_reason = ""

        if items_data and len(items_data) > 0:
            first_item = items_data[0]
            jd_id = first_item.get('jd_id', '')
            if 'source_phrase' in first_item:
                source_phrase = first_item.get('source_phrase', '')
            if 'reason' in first_item:
                existing_reason = first_item.get('reason', '')

        items_to_classify.append({
            'item': row[0],
            'definition': row[1] or '',
            'source': row[2] or CONFIG['merged_table'],
            'inserted_timestamp': row[3],
            'existing_reason': existing_reason,
            'source_phrase': source_phrase,
            'jd_id': jd_id,
            'domain': row[5] or ''
        })

    print(f"   ✅ Found {len(items_to_classify)} unclassified {CATEGORY.lower()} from merged data")
    return items_to_classify


# ============================================================================
# FUZZY MATCHING AND CATEGORIZATION
# ============================================================================
def find_best_match_in_existing(new_item, existing_items, threshold=85):
    """Find the best matching existing item"""
    best_similarity = 0
    best_match = None
    match_type = None
    matched_item = None

    for existing in existing_items:
        # Check similarity with main item field
        main_similarity = calculate_similarity(new_item['item'], existing['item'])

        if main_similarity > best_similarity:
            best_similarity = main_similarity
            best_match = existing
            match_type = f'MAIN_{CONFIG["item_column"]}'
            matched_item = None

        # Check similarity with each item in data JSON
        if existing['items_data']:
            for data_item in existing['items_data']:
                item_value = data_item.get(CONFIG['json_item_key'], '')
                if item_value:
                    item_similarity = calculate_similarity(new_item['item'], item_value)

                    if item_similarity > best_similarity:
                        best_similarity = item_similarity
                        best_match = existing
                        match_type = f'{CONFIG["data_json_column"]}_ITEM'
                        matched_item = item_value

    if best_similarity >= threshold:
        return best_match, best_similarity, match_type, matched_item

    return None, best_similarity, None, None


def find_duplicate_groups_in_new_items(new_items, threshold=85):
    print(f"\n🔍 Finding duplicate groups within new {CATEGORY.lower()} (threshold: {threshold}%)...")
    sim_overall_start = time.time()

    if not new_items:
        return []

    n = len(new_items)
    print(f"   📊 Total items to compare: {n} ({n * (n - 1) // 2} pairs)")

    # ── PARALLEL SIMILARITY COMPUTATION ──────────────────────────────────────
    # Split row-pairs across threads for faster computation
    print(f"   ⚡ Computing pairwise similarities IN PARALLEL...")
    compute_start = time.time()

    def compute_row_similarities(row_idx):
        """Compute similarities for all pairs where i == row_idx"""
        matches = []
        for j in range(row_idx + 1, n):
            sim = calculate_similarity(new_items[row_idx]['item'], new_items[j]['item'])
            if sim >= threshold:
                matches.append((row_idx, j, sim))
        return matches

    all_matches = []
    max_workers = min(8, n)  # Cap threads at 8 to avoid overhead for small n
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(compute_row_similarities, i) for i in range(n)]
        for future in as_completed(futures):
            all_matches.extend(future.result())

    compute_elapsed = time.time() - compute_start
    print(f"   ✅ Similarity computation completed in {compute_elapsed:.2f}s — {len(all_matches)} matches found")

    # ── UNION-FIND ────────────────────────────────────────────────────────────
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for i, j, sim in all_matches:
        union(i, j)
        print(f"   🔗 Matched: '{new_items[i]['item']}' <-> '{new_items[j]['item']}' (similarity: {sim}%)")

    # ── GROUP BY ROOT ─────────────────────────────────────────────────────────
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    groups = []
    for indices in groups_dict.values():
        group = []
        for i in indices:
            if len(indices) == 1:
                group.append((new_items[i], 100))
            else:
                max_sim = 100
                for j in indices:
                    if i != j:
                        sim = calculate_similarity(new_items[i]['item'], new_items[j]['item'])
                        max_sim = max(max_sim, sim)
                group.append((new_items[i], max_sim))
        groups.append(group)

    duplicate_groups = [g for g in groups if len(g) > 1]
    singleton_groups = [g for g in groups if len(g) == 1]

    sim_overall_elapsed = time.time() - sim_overall_start
    print(f"\n   📊 Grouping results:")
    print(f"      - Total groups: {len(groups)}")
    print(f"      - Groups with 2+ items: {len(duplicate_groups)}")
    print(f"      - Singleton groups: {len(singleton_groups)}")
    print(f"      - Total items processed: {n}")
    print(f"      - Total pairwise matches found: {len(all_matches)}")
    print(f"      ⏱️  Total similarity time: {sim_overall_elapsed:.2f}s")

    if duplicate_groups:
        print(f"\n   📦 Multi-item group details:")
        for idx, group in enumerate(duplicate_groups, 1):
            items_in_group = [item[0]['item'] for item in group]
            print(f"      Group {idx} ({len(group)} items): {', '.join(items_in_group)}")

    return groups


def select_best_item_from_group(items_group):
    """
    Select the best representative item from a group.
    Prefers items with existing reasons, then longest name.
    """
    items_with_reason = [t for t in items_group if t.get('existing_reason')]
    if items_with_reason:
        return max(items_with_reason, key=lambda t: len(t['item']))

    return max(items_group, key=lambda t: len(t['item']))


def categorize_new_items(new_items, existing_items, threshold=85):
    """
    NEW LOGIC: Categorize new items into two groups:
    1. Items matching existing records (to add to existing)
    2. Items not matching existing (will be grouped among themselves)
    """
    print(f"\n{'=' * 70}")
    print(f"🔍 CATEGORIZING NEW {CATEGORY} (Threshold: {threshold}%)")
    print(f"{'=' * 70}")

    to_add_to_existing = []
    not_matching_existing = []

    # PHASE 1: Compare new items with existing merged records
    print(f"\n📊 PHASE 1: Comparing {len(new_items)} new items with {len(existing_items)} existing records...")

    for idx, new_item in enumerate(new_items, 1):
        print(f"\n{'─' * 70}")
        print(f"🔍 Analyzing [{idx}/{len(new_items)}]: {new_item['item']}")

        best_match, similarity, match_type, matched_item = find_best_match_in_existing(
            new_item, existing_items, threshold
        )

        if best_match:
            print(f"   ✅ MATCH FOUND: {similarity}% with '{best_match['item']}'")
            print(f"   📍 Match type: {match_type}")
            if matched_item:
                print(f"   📝 Matched item: '{matched_item}'")
            print(f"   ➕ Action: ADD to existing record")

            to_add_to_existing.append({
                'new_item': new_item,
                'existing_record': best_match,
                'similarity': similarity,
                'match_type': match_type,
                'matched_item': matched_item
            })
        else:
            print(f"   ❌ NO MATCH: Highest similarity {similarity}% (below threshold)")
            print(f"   📦 Action: Will group with other unmatched items")

            not_matching_existing.append(new_item)

    # PHASE 2: Group items that didn't match existing records among themselves
    print(f"\n{'=' * 70}")
    print(f"📊 PHASE 2: Grouping {len(not_matching_existing)} unmatched items among themselves...")
    print(f"{'=' * 70}")

    to_create_new = []

    if not_matching_existing:
        # Perform fuzzy matching within unmatched items
        groups = find_duplicate_groups_in_new_items(not_matching_existing, threshold)

        print(f"\n📦 Creating records from {len(groups)} groups...")
        for group_idx, group in enumerate(groups, 1):
            # Select best representative from the group
            best_item = select_best_item_from_group([item[0] for item in group])

            # Create items_data JSON array from all items in the group
            items_data = []
            for item, similarity in group:
                items_data.append({
                    CONFIG['json_item_key']: item['item'],
                    'definition': item['definition'],
                    'source_table': item['source'],
                    'fuzzy_similarity': similarity,
                    'jd_id': item.get('jd_id', ''),
                    'domain': item.get('domain', '')
                })

            to_create_new.append({
                'new_item': best_item,
                'best_similarity': 100 if len(group) > 1 else 0,
                'grouped_items': items_data
            })

            if len(group) > 1:
                item_names = [t[0]['item'] for t in group if t[0] != best_item]
                print(f"   🔗 Group {group_idx}: '{best_item['item']}' (selected) ← {item_names}")
            else:
                print(f"   📌 Singleton {group_idx}: '{best_item['item']}' (no similar items)")

    # SUMMARY
    print(f"\n{'=' * 70}")
    print(f"📊 CATEGORIZATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"   Total new {CATEGORY.lower()} analyzed: {len(new_items)}")
    print(f"   {'─' * 66}")
    print(
        f"   🔗 Matching existing (to add): {len(to_add_to_existing)} ({len(to_add_to_existing) / len(new_items) * 100:.1f}%)")
    print(
        f"   ❌ Not matching existing: {len(not_matching_existing)} ({len(not_matching_existing) / len(new_items) * 100:.1f}%)")
    print(f"   {'─' * 66}")
    print(f"   📦 Groups created from unmatched: {len(to_create_new)}")
    print(f"      - Groups with 2+ items: {sum(1 for g in to_create_new if len(g['grouped_items']) > 1)}")
    print(f"      - Singleton groups: {sum(1 for g in to_create_new if len(g['grouped_items']) == 1)}")
    print(f"   {'─' * 66}")

    return {
        'to_add_to_existing': to_add_to_existing,
        'to_create_new': to_create_new
    }


# ============================================================================
# DATABASE UPDATE FUNCTIONS
# ============================================================================
def update_existing_records_with_additions(cur, conn, to_add_to_existing):
    """Update existing records by adding new items to their data JSON array"""
    if not to_add_to_existing:
        print(f"\n⏭️  No records to update with additions")
        return 0

    print(f"\n{'=' * 70}")
    print(f"💾 UPDATING EXISTING RECORDS WITH NEW {CATEGORY}")
    print(f"{'=' * 70}")

    updates_by_record = {}
    for item in to_add_to_existing:
        existing_item = item['existing_record']['item']
        if existing_item not in updates_by_record:
            updates_by_record[existing_item] = {
                'existing_record': item['existing_record'],
                'new_items': []
            }
        updates_by_record[existing_item]['new_items'].append({
            'item_data': item['new_item'],
            'similarity': item['similarity']
        })

    updated_count = 0

    for existing_item, data in updates_by_record.items():
        print(f"\n📝 Updating: {existing_item}")
        print(f"   Current {CONFIG['data_json_column']} items: {len(data['existing_record']['items_data'])}")
        print(f"   Adding {len(data['new_items'])} new item(s):")

        current_items_data = data['existing_record']['items_data'].copy()

        for new_item_info in data['new_items']:
            new_item = new_item_info['item_data']
            similarity = new_item_info['similarity']

            new_data_item = {
                CONFIG['json_item_key']: new_item['item'],
                'definition': new_item['definition'],
                'source_table': new_item['source'],
                'fuzzy_similarity': similarity,
                'jd_id': new_item.get('jd_id', ''),
                'domain': new_item.get('domain', '')
            }
            current_items_data.append(new_data_item)
            print(
                f"      + '{new_item['item']}' (similarity: {similarity}%, JD_ID: {new_item.get('jd_id', 'N/A')}, Domain: {new_item.get('domain', 'N/A')})")

        jd_domain = extract_distinct_domains(current_items_data)

        update_query = f"""
        UPDATE "{SCHEMA}"."{CONFIG['merged_table']}"
        SET "{CONFIG['data_json_column']}" = %s,
            "JD_DOMAIN" = %s,
            "UPDATED_TIMESTAMP" = %s
        WHERE "{CONFIG['item_column']}" = %s
        """

        try:
            cur.execute(
                update_query,
                (Json(current_items_data), jd_domain, datetime.now(), existing_item)
            )

            if cur.rowcount > 0:
                updated_count += 1
                print(f"   ✅ Updated successfully")
                print(f"   📊 New {CONFIG['data_json_column']} count: {len(current_items_data)}")
                print(f"   🏷️ JD_DOMAIN: {jd_domain}")
            else:
                print(f"   ⚠️  Update failed - no rows affected")

        except Exception as e:
            print(f"   ❌ Error updating record: {e}")

    conn.commit()

    print(f"\n{'=' * 70}")
    print(f"✅ UPDATE COMPLETED")
    print(f"   Records updated: {updated_count}")
    print(f"   Total {CATEGORY.lower()} added: {len(to_add_to_existing)}")
    print(f"{'=' * 70}")

    return updated_count


def insert_new_records(cur, conn, to_create_new):
    """Insert new records into merged table"""
    if not to_create_new:
        print(f"\n⏭️  No new records to insert")
        return 0

    print(f"\n{'=' * 70}")
    print(f"💾 INSERTING NEW RECORDS")
    print(f"{'=' * 70}")

    insert_query = f"""
    INSERT INTO "{SCHEMA}"."{CONFIG['merged_table']}" (
        "{CONFIG['item_column']}",
        "DEFINITION",
        "SOURCE",
        "INSERTED_TIMESTAMP",
        "{CONFIG['data_json_column']}",
        "JD_DOMAIN"
    )
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT ("{CONFIG['item_column']}") DO NOTHING
    """

    inserted_count = 0

    for idx, item in enumerate(to_create_new, 1):
        new_item = item['new_item']
        items_data = item.get('grouped_items', [])

        jd_domain = extract_distinct_domains(items_data)

        print(f"\n📝 [{idx}/{len(to_create_new)}] Inserting: {new_item['item']}")
        if len(items_data) > 1:
            print(f"   🔗 Contains {len(items_data)} grouped {CATEGORY.lower()}:")
            for td in items_data:
                print(
                    f"      - {td[CONFIG['json_item_key']]} (JD_ID: {td.get('jd_id', 'N/A')}, Domain: {td.get('domain', 'N/A')})")
        else:
            print(f"   📋 JD_ID: {new_item.get('jd_id', 'N/A')}, Domain: {new_item.get('domain', 'N/A')}")
        print(f"   🏷️ JD_DOMAIN: {jd_domain}")

        try:
            cur.execute(
                insert_query,
                (
                    new_item['item'],
                    new_item['definition'],
                    new_item['source'],
                    datetime.now(),
                    Json(items_data),
                    jd_domain
                )
            )

            if cur.rowcount > 0:
                inserted_count += 1
                print(f"   ✅ Inserted successfully")
            else:
                print(f"   ⏭️  Skipped (already exists)")

        except Exception as e:
            print(f"   ❌ Error inserting: {e}")

    conn.commit()

    print(f"\n{'=' * 70}")
    print(f"✅ INSERTION COMPLETED")
    print(f"   New records inserted: {inserted_count}")
    print(f"{'=' * 70}")

    return inserted_count


# ============================================================================
# LLM CLASSIFICATION FUNCTIONS
# ============================================================================
def get_llm_classification(item_name, definition, existing_reason, api_key):
    """Call LLM to classify an item"""

    existing_reason_context = ""
    if existing_reason and existing_reason.strip():
        existing_reason_context = f"""
    **Existing Classification Reason**: "{existing_reason}"

    NOTE: An existing reason has been provided above. Please validate whether this reason correctly identifies the category. If the existing reason is incorrect or the item belongs to a different category, provide the correct classification with an updated reason explaining why."""

    template = """
    I have one item with the following details:

    **Name**: "{item_name}"
    **Definition**: "{definition}"
    {existing_reason_context}

    Classify it across the following six categories using Yes/No flags:

    - IS_TOOL: "Yes" if it's a tool, software, physical instrument, platform, or framework used to perform tasks related to design, development, analysis, or testing of a product.

    - IS_TECHNOLOGY: "Yes" if it's a technology, technical concept, programming language, system, method, algorithm, or innovation that enables product development or functionality.

    - IS_COMPLIANCE: "Yes" if it's related to compliance, regulation, law, standard, certification, or policy required by authorities or industry bodies.

    - IS_METHODOLOGY: "Yes" if it's a process, practice, or structured approach (e.g., Agile, V-model, Scrum, Six Sigma).

    - IS_PRODUCT: "Yes" if it's a product, platform, hardware, or solution provided by a company.

    - IS_IRRELEVANT: "Yes" if it does not belong to any of the above categories.

    ### Additional Requirements:
    - Exactly one of the six categories must be "Yes".
    - Provide a **REASON** explaining why the item belongs to that category.
    - If an existing reason was provided, validate it and either confirm it's correct or explain why it's incorrect and provide the correct classification.
    - Provide a **CONFIDENCE_SCORE** between 0 and 1 (e.g., 0.95).

    ### Output format:
    Return a markdown table with columns:

    | EXTRACTED_NAME | IS_TOOL | IS_TECHNOLOGY | IS_COMPLIANCE | IS_METHODOLOGY | IS_PRODUCT | IS_IRRELEVANT | REASON | CONFIDENCE_SCORE |
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key, temperature=LLM_TEMPERATURE)
    chain = LLMChain(prompt=prompt, llm=llm)

    try:
        response = chain.run(
            item_name=item_name,
            definition=definition,
            existing_reason_context=existing_reason_context
        )
        return response
    except Exception as e:
        print(f"   ❌ LLM Error for '{item_name}': {e}")
        return None


def parse_llm_response(item_name, response):
    """Parse LLM response table"""
    if not response:
        return None

    lines = [l.strip() for l in response.split("\n") if "|" in l and "---" not in l]

    if len(lines) < 2:
        print(f"   ⚠️ Parsing failed for {item_name} → No valid table")
        return None

    parts = [p.strip().strip('"').strip("'") for p in lines[1].split("|") if p.strip()]

    if len(parts) < 9:
        print(f"   ⚠️ Parsing failed for {item_name} → Malformed format")
        return None

    return {
        "EXTRACTED_NAME": parts[0],
        "IS_TOOL": parts[1],
        "IS_TECHNOLOGY": parts[2],
        "IS_COMPLIANCE": parts[3],
        "IS_METHODOLOGY": parts[4],
        "IS_PRODUCT": parts[5],
        "IS_IRRELEVANT": parts[6],
        "REASON": parts[7],
        "CONFIDENCE_SCORE": parts[8]
    }


def classify_item_with_llm(item_record, api_key):
    """Classify a single item using LLM"""
    item_name = item_record['item']
    definition = item_record['definition']
    existing_reason = item_record.get('existing_reason', '')
    jd_id = item_record.get('jd_id', '')
    source_phrase = item_record.get('source_phrase', '')

    print(f"\n🔵 Classifying: {item_name}")
    if existing_reason:
        print(f"   📝 Has existing reason: {existing_reason[:80]}{'...' if len(existing_reason) > 80 else ''}")
    if jd_id:
        print(f"   🔗 JD_ID: {jd_id}")

    start_time = time.time()

    try:
        response = get_llm_classification(item_name, definition, existing_reason, api_key)
        parsed = parse_llm_response(item_name, response)

        if parsed:
            parsed['SOURCE'] = item_record['source']
            parsed['DESCRIPTION'] = definition
            parsed['ORIGINAL_REASON'] = existing_reason
            parsed['REFERENCE_ID'] = jd_id
            parsed['SOURCE_DATA'] = source_phrase

        elapsed = round(time.time() - start_time, 2)

        if parsed:
            category = get_category_from_parsed(parsed)
            print(f"🟢 Completed: {item_name} - {elapsed}s - Category: {category}")
        else:
            print(f"🟡 Completed: {item_name} - {elapsed}s - Failed to parse response")

        return parsed

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"🔴 Error: {item_name} - {elapsed}s - {str(e)[:100]}")
        return None


def get_category_from_parsed(parsed):
    """Helper function to extract the category that was marked as 'Yes'"""
    categories = ['IS_TOOL', 'IS_TECHNOLOGY', 'IS_COMPLIANCE', 'IS_METHODOLOGY', 'IS_PRODUCT', 'IS_IRRELEVANT']
    for cat in categories:
        if parsed.get(cat, '').lower() == 'yes':
            return cat.replace('IS_', '')
    return 'UNKNOWN'


def insert_llm_classifications_batch(cur, conn, classifications, batch_num):
    """Insert LLM classification results for a single batch"""
    if not classifications:
        print(f"   ⚠️  Batch {batch_num}: No valid classifications to insert")
        return 0

    print(f"\n💾 Batch {batch_num}: Inserting {len(classifications)} classifications...")

    query = f"""
        INSERT INTO "{SCHEMA}"."{PIPELINE_TABLES['LLM_CLASSIFICATION']}"
        ("EXTRACTED_NAME","IS_TOOL","IS_TECHNOLOGY","IS_COMPLIANCE",
         "IS_METHODOLOGY","IS_PRODUCT","IS_IRRELEVANT",
         "REASON","CONFIDENCE_SCORE","DESCRIPTION","INSERTED_TIMESTAMP","SOURCE",
         "REFERENCE_ID","SOURCE_DATA")
        VALUES %s
        ON CONFLICT ("EXTRACTED_NAME") DO NOTHING;
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data = [
        (
            c["EXTRACTED_NAME"],
            c["IS_TOOL"],
            c["IS_TECHNOLOGY"],
            c["IS_COMPLIANCE"],
            c["IS_METHODOLOGY"],
            c["IS_PRODUCT"],
            c["IS_IRRELEVANT"],
            c["REASON"],
            c["CONFIDENCE_SCORE"],
            c.get("DESCRIPTION", ""),
            timestamp,
            c["SOURCE"],
            c.get("REFERENCE_ID", ""),
            c.get("SOURCE_DATA", "")
        )
        for c in classifications if c is not None
    ]

    try:
        execute_values(cur, query, data)
        conn.commit()
        print(f"   ✅ Batch {batch_num}: Inserted {len(data)} classifications")
        return len(data)
    except Exception as e:
        print(f"   ❌ Batch {batch_num}: Error inserting classifications: {e}")
        conn.rollback()
        return 0


def process_batch_with_llm(batch_records, batch_num, api_key, cur, conn):
    """Process a single batch: classify all items IN PARALLEL then insert"""
    print(f"\n{'=' * 70}")
    print(f"📦 PROCESSING LLM BATCH {batch_num}")
    print(f"   Items in batch: {len(batch_records)}")
    print(f"   ⚡ All {len(batch_records)} LLM calls will fire IN PARALLEL")
    print(f"{'=' * 70}")

    batch_start_time = time.time()

    # ── FIRE ALL LLM CALLS IN PARALLEL ───────────────────────────────────────
    llm_start = time.time()

    async def run_batch_llm_async(records, key):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=len(records)) as executor:
            tasks = [
                loop.run_in_executor(executor, classify_item_with_llm, record, key)
                for record in records
            ]
            return await asyncio.gather(*tasks)

    raw_results = asyncio.run(run_batch_llm_async(batch_records, api_key))
    llm_elapsed = time.time() - llm_start
    print(f"\n   ⚡ All parallel LLM calls completed in {llm_elapsed:.2f}s")

    # ── COLLECT RESULTS ───────────────────────────────────────────────────────
    batch_classifications = []
    failed_count = 0

    for idx, (record, classification) in enumerate(zip(batch_records, raw_results), 1):
        if classification:
            batch_classifications.append(classification)
            print(f"   ✅ [{idx}/{len(batch_records)}] {record['item'][:40]} → SUCCESS")
        else:
            failed_count += 1
            print(f"   ❌ [{idx}/{len(batch_records)}] {record['item'][:40]} → FAILED")

    batch_elapsed = time.time() - batch_start_time

    print(f"\n{'─' * 70}")
    print(f"📊 BATCH {batch_num} COMPLETED")
    print(f"   ⏱️  Total batch time:       {batch_elapsed:.2f}s")
    print(f"   ⚡ LLM parallel call time: {llm_elapsed:.2f}s")
    print(f"   ✅ Successful:             {len(batch_classifications)}")
    print(f"   ❌ Failed:                 {failed_count}")
    print(f"   ⚡ Avg per item (parallel): {llm_elapsed / len(batch_records):.2f}s effective")
    print(f"{'─' * 70}")

    # ── INSERT AFTER BATCH COMPLETES ──────────────────────────────────────────
    inserted = insert_llm_classifications_batch(cur, conn, batch_classifications, batch_num)

    return len(batch_classifications), failed_count, inserted


# ============================================================================
# LOGGING
# ============================================================================
def insert_log(cur, action, table_name, new_count, existing_count, total_count):
    """Insert log entry"""
    print(f"\n📝 Inserting log into TTC_MASTER_DATA_LOGS...")

    cur.execute("""
        SELECT COALESCE(MAX(CAST(SUBSTRING("LOG_ID" FROM 'LOG_(\\d+)') AS INTEGER)), 0) + 1
        FROM "LOGS"."PROCESSED_MASTER_DATA_LOGS"
    """)
    next_id = cur.fetchone()[0]
    log_id = f"LOG_{next_id:07d}"

    insert_query = """
    INSERT INTO "LOGS"."PROCESSED_MASTER_DATA_LOGS" (
        "LOG_ID",
        "ACTION",
        "TABLE",
        "NEW_COUNT",
        "EXISTING_COUNT",
        "TOTAL_COUNT",
        "INSERTED_TIMESTAMP"
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cur.execute(
        insert_query,
        (
            log_id,
            action,
            table_name,
            new_count,
            existing_count,
            total_count,
            datetime.now()
        )
    )

    print(f"   ✅ Log inserted: {log_id}")
    return log_id


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================
def merge_and_classify_pipeline():
    global RUN_LLM_CLASSIFICATION
    """
    Main function with improved fuzzy matching logic:
    1. Match new items with existing records → add to existing
    2. Group remaining unmatched items among themselves → create new records
    3. Classify all unclassified items with LLM
    """

    print("=" * 70)
    print(f"🚀 {CATEGORY} MERGER + LLM CLASSIFICATION (IMPROVED FUZZY MATCHING)")
    print("=" * 70)

    conn, cur = connect_to_db()

    try:
        overall_start = time.time()

        # STEP 1: Fetch all data
        print("\n" + "=" * 70)
        print("STEP 1: DATA FETCHING")
        print("=" * 70)

        max_timestamp = get_max_timestamp(cur)
        existing_items = fetch_existing_merged_items(cur)
        items_from_standardization = fetch_new_items_from_standardization(cur, max_timestamp)
        items_from_masterdata = fetch_new_items_from_masterdata(cur, max_timestamp)

        all_new_items = items_from_standardization + items_from_masterdata

        print(f"\n📊 Data Summary:")
        print(f"   Existing merged {CATEGORY.lower()}: {len(existing_items)}")
        print(f"   New {CATEGORY.lower()} to process: {len(all_new_items)}")

        # STEP 2: Categorize new items (only if there are new items)
        updated_count = 0
        inserted_count = 0

        if all_new_items:
            print("\n" + "=" * 70)
            print("STEP 2: FUZZY MATCHING & CATEGORIZATION")
            print("=" * 70)

            categorization = categorize_new_items(all_new_items, existing_items, FUZZY_SIMILARITY_THRESHOLD)
            to_add_to_existing = categorization['to_add_to_existing']
            to_create_new = categorization['to_create_new']

            # STEP 3: Update existing records
            print("\n" + "=" * 70)
            print("STEP 3: UPDATE EXISTING RECORDS")
            print("=" * 70)

            updated_count = update_existing_records_with_additions(cur, conn, to_add_to_existing)

            # STEP 4: Insert new records
            print("\n" + "=" * 70)
            print("STEP 4: INSERT NEW RECORDS")
            print("=" * 70)

            inserted_count = insert_new_records(cur, conn, to_create_new)

            # Log merging operation
            merge_log_id = insert_log(
                cur,
                action=f"{CATEGORY}_MERGED_WITH_IMPROVED_FUZZY_MATCHING",
                table_name=CONFIG['merged_table'],
                new_count=inserted_count,
                existing_count=updated_count,
                total_count=len(all_new_items)
            )
            conn.commit()
        else:
            print(f"\n⚠️  No new {CATEGORY.lower()} to process from source tables")
            merge_log_id = None

        # STEP 5: LLM Classification (for ALL unclassified items in merged data)
        # STEP 5: LLM Classification (for ALL unclassified items in merged data)
        print("\n" + "=" * 70)
        print(f"STEP 5: LLM CLASSIFICATION (ALL UNCLASSIFIED {CATEGORY})")
        print("=" * 70)

        total_successful = 0
        total_failed = 0
        total_inserted_llm = 0
        unclassified_items = []
        llm_log_id = None


        if not RUN_LLM_CLASSIFICATION:
            print(f"\n⏭️  SKIPPED — RUN_LLM_CLASSIFICATION = False")
            print(f"   💡 Set RUN_LLM_CLASSIFICATION = True to run classification")
            print(f"   📦 Merging is complete. Unclassified items are waiting in {CONFIG['merged_table']}")
        else:
            unclassified_items = fetch_unclassified_items_from_merged_data(cur)

            if unclassified_items:
                print(
                    f"\n🤖 Starting LLM classification for {len(unclassified_items)} unclassified {CATEGORY.lower()}...")
                print(f"   Batch size: {LLM_BATCH_SIZE}")
                print(f"   Total batches: {(len(unclassified_items) + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE}")

                for batch_start in range(0, len(unclassified_items), LLM_BATCH_SIZE):
                    batch_end = min(batch_start + LLM_BATCH_SIZE, len(unclassified_items))
                    batch_records = unclassified_items[batch_start:batch_end]
                    batch_num = (batch_start // LLM_BATCH_SIZE) + 1

                    successful, failed, inserted = process_batch_with_llm(
                        batch_records,
                        batch_num,
                        OPENAI_API_KEY,
                        cur,
                        conn
                    )

                    total_successful += successful
                    total_failed += failed
                    total_inserted_llm += inserted

                if total_successful > 0:
                    llm_log_id = insert_log(
                        cur,
                        action=f"{CATEGORY}_LLM_CLASSIFICATION_FROM_MERGED_DATA",
                        table_name="NEW_TTC_LLM_CLASSIFICATION",
                        new_count=total_successful,
                        existing_count=total_failed,
                        total_count=len(unclassified_items)
                    )
                    conn.commit()
            else:
                print(f"\n✅ All {CATEGORY.lower()} in merged data are already classified!")

        # FINAL SUMMARY
        total_elapsed = time.time() - overall_start

        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print(f"\n📊 MERGING SUMMARY:")
        if all_new_items:
            print(f"   {'─' * 66}")
            print(f"   Total new {CATEGORY.lower()} processed: {len(all_new_items)}")
            print(f"   {'─' * 66}")
            print(f"   🔗 Matched with existing: {len(to_add_to_existing)}")
            print(f"      Records updated: {updated_count}")
            print(f"   {'─' * 66}")
            print(f"   📦 Grouped among themselves: {len(to_create_new)}")
            print(f"      Records inserted: {inserted_count}")
        else:
            print(f"   No new {CATEGORY.lower()} to merge")

        print(f"\n📊 LLM CLASSIFICATION SUMMARY:")
        print(f"   {'─' * 66}")
        print(f"   Total unclassified {CATEGORY.lower()}: {len(unclassified_items) if unclassified_items else 0}")
        print(f"   LLM classifications successful: {total_successful}")
        print(f"   LLM failures: {total_failed}")
        print(f"   Records inserted to classification table: {total_inserted_llm}")
        print(f"   {'─' * 66}")
        print(f"\n⏱️  Total time: {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")
        if merge_log_id:
            print(f"   📝 Merge log ID: {merge_log_id}")
        if llm_log_id:
            print(f"   📝 LLM log ID: {llm_log_id}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()

    finally:
        cur.close()
        conn.close()
        print("\n✅ Database connection closed")

# # # ============================================================================
# # ENTRY POINT
# # ============================================================================
# if __name__ == "__main__":
#     merge_and_classify_pipeline()
