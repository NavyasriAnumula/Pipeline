import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import json
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

from config import (
    DB_CONFIG, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL,
    OPENAI_API_KEY_GPT,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SCHEMA,
    FUZZY_SIMILARITY_THRESHOLD,
    DEFAULT_MIN_TIMESTAMP,
    CATEGORY_CONFIG,
    PIPELINE_TABLES
)

SCHEMA = "PROCESSING"

# Batch configuration
BATCH_SIZE = 6  # Records per batch for database commit
LLM_BATCH_SIZE = 3  # Items to send to LLM in one API call

# Initialize DeepSeek client
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

labeled_table = PIPELINE_TABLES["LABELED"]

# ============================================================================
# SET CATEGORY HERE - CHANGE THIS TO RUN DIFFERENT PIPELINES
# ============================================================================
CATEGORY = "COMPLIANCES"  # Options: "TOOLS", "TECHNOLOGIES", "COMPLIANCES"

# Get configuration for selected category
CONFIG = CATEGORY_CONFIG[CATEGORY]


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
# DEEPSEEK API CALL
# ============================================================================

def call_deepseek(prompt, retries=2):
    """
    Call DeepSeek API with retry logic

    Args:
        prompt: The prompt to send to DeepSeek
        retries: Number of retries on failure

    Returns:
        API response content as string
    """
    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that validates technical classification data. You must respond ONLY with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False
        )

        elapsed = time.time() - start_time
        content = response.choices[0].message.content.strip()

        return content, elapsed

    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0

        if retries > 0:
            print(f"      ⚠️  API Error, retrying... ({retries} left)")
            time.sleep(2)
            return call_deepseek(prompt, retries - 1)

        print(f"      ❌ API Error: {str(e)[:100]}")
        return None, elapsed


# ============================================================================
# DATA FETCHING AND GROUPING
# ============================================================================

def fetch_category_records(cur, category):
    """
    Fetch records for a specific category

    Args:
        cur: Database cursor
        category: 'TOOL', 'TECHNOLOGY', or 'COMPLIANCE'

    Returns:
        List of records for the category
    """
    category_field_map = {
        'TOOL': 'IS_TOOL',
        'TECHNOLOGY': 'IS_TECHNOLOGY',
        'COMPLIANCE': 'IS_COMPLIANCE'
    }

    field = category_field_map.get(category)
    if not field:
        return []

    query = f"""
    SELECT 
        "EXTRACTED_NAME",
        "IS_TOOL",
        "IS_TECHNOLOGY",
        "IS_COMPLIANCE",
        "IS_METHODOLOGY",
        "IS_PRODUCT",
        "IS_IRRELEVANT",
        "REASON",
        "CONFIDENCE_SCORE",
        "DESCRIPTION",
        "SOURCE"
    FROM "PROCESSING"."MASTERDATA_LLM_CLASSIFICATION_VALIDATION"
    WHERE "IS_VALID" IS NULL 
    AND lower("{field}") = 'yes'
    ORDER BY "INSERTED_TIMESTAMP" DESC

    """

    cur.execute(query)
    rows = cur.fetchall()

    records = []
    for row in rows:
        records.append({
            'extracted_name': row[0],
            'is_tool': row[1],
            'is_technology': row[2],
            'is_compliance': row[3],
            'is_methodology': row[4],
            'is_product': row[5],
            'is_irrelevant': row[6],
            'reason': row[7] or '',
            'confidence_score': row[8] or '',
            'description': row[9] or '',
            'source': row[10] or ''
        })

    return records


def fetch_non_validated_other_categories(cur):
    """
    Fetch records marked as METHODOLOGY, PRODUCT, or IRRELEVANT that need direct insertion

    Args:
        cur: Database cursor

    Returns:
        List of records with their category
    """
    query = """
   SELECT  
        "EXTRACTED_NAME",
        "DESCRIPTION",
        "IS_METHODOLOGY",
        "IS_PRODUCT",
        "IS_IRRELEVANT"
    FROM  "PROCESSING"."MASTERDATA_LLM_CLASSIFICATION_VALIDATION"
    WHERE "IS_VALID" IS NULL 
    AND (
        lower("IS_METHODOLOGY") = 'yes' OR
        lower("IS_PRODUCT") = 'yes' OR
        lower("IS_IRRELEVANT") = 'yes'
    )
    ORDER BY "INSERTED_TIMESTAMP" DESC
    """

    cur.execute(query)
    rows = cur.fetchall()

    records = []
    for row in rows:
        extracted_name = row[0]
        description = row[1] or ''
        is_methodology = row[2]
        is_product = row[3]
        is_irrelevant = row[4]

        # Determine category
        if is_methodology and is_methodology.lower() == 'yes':
            category = 'METHODOLOGY'
        elif is_product and is_product.lower() == 'yes':
            category = 'PRODUCT'
        elif is_irrelevant and is_irrelevant.lower() == 'yes':
            category = 'IRRELEVANT'
        else:
            continue

        records.append({
            'extracted_name': extracted_name,
            'description': description,
            'category': category
        })

    return records


# ============================================================================
# CLASSIFIED DATA INSERT FUNCTIONS
# ============================================================================

def insert_product(cur, extracted_name, definition):
    """Insert product into CLASSIFIED_PRODUCTS_DATA"""
    query = """
    INSERT INTO "PROCESSING"."CLASSIFIED_PRODUCTS_DATA" (
        "PRODUCT",
        "DEFINITION",
        "VALIDATED_BY",
        "INSERTED_TIMESTAMP"
    )
    VALUES (%s, %s, %s, NOW())
    ON CONFLICT ("PRODUCT") DO NOTHING;
    """
    cur.execute(query, (extracted_name, definition, 'AUTO_CLASSIFIED'))


def insert_methodology(cur, extracted_name, definition):
    """Insert methodology into CLASSIFIED_METHODOLOGIES_DATA"""
    query = """
    INSERT INTO "PROCESSING"."CLASSIFIED_METHODOLOGIES_DATA" (
        "METHODOLOGY",
        "DEFINITION",
        "VALIDATED_BY",
        "INSERTED_TIMESTAMP"
    )
    VALUES (%s, %s, %s, NOW())
    ON CONFLICT ("METHODOLOGY") DO NOTHING;
    """
    cur.execute(query, (extracted_name, definition, 'AUTO_CLASSIFIED'))


def insert_irrelevant(cur, extracted_name, definition):
    """Insert irrelevant extracted_name into CLASSIFIED_IRRELEVANT_DATA"""
    query = """
    INSERT INTO "PROCESSING"."CLASSIFIED_IRRELEVANT_DATA" (
        "WORD",
        "DEFINITION",
        "VALIDATED_BY",
        "INSERTED_TIMESTAMP"
    )
    VALUES (%s, %s, %s, NOW())
    ON CONFLICT ("WORD") DO NOTHING;
    """
    cur.execute(query, (extracted_name, definition, 'AUTO_CLASSIFIED'))


def handle_classified_insert(cur, category, extracted_name, definition):
    """
    Route records to appropriate classified tables

    Args:
        cur: Database cursor
        category: Category type (PRODUCT, METHODOLOGY, IRRELEVANT)
        extracted_name: The extracted_name/term
        definition: Description
    """
    if category == "PRODUCT":
        insert_product(cur, extracted_name, definition)
    elif category == "METHODOLOGY":
        insert_methodology(cur, extracted_name, definition)
    elif category == "IRRELEVANT":
        insert_irrelevant(cur, extracted_name, definition)


def mark_as_processed(cur, extracted_name, category):
    update_query = """
    UPDATE "PROCESSING"."MASTERDATA_LLM_CLASSIFICATION_VALIDATION"
    SET 
        "IS_VALID" = %s,
        "UPDATED_TIMESTAMP" = %s,
        "UPDATED_CATEGORY" = %s,
        "VALIDATED_BY" = %s,
        "COMMENTS" = %s
    WHERE "EXTRACTED_NAME" = %s
    """

    cur.execute(
        update_query,
        (
            'AUTO_CLASSIFIED',  # IS_VALID
            datetime.now(),  # UPDATED_TIMESTAMP
            category,  # UPDATED_CATEGORY
            'AUTO_CLASSIFIED',  # VALIDATED_BY
            f'Automatically classified as {category} - no LLM needed',  # COMMENTS
            extracted_name  # WHERE clause
        )
    )


# ============================================================================
# PROCESS OTHER CATEGORIES (NO LLM VALIDATION)
# ============================================================================

def process_other_categories(cur, conn):
    """
    Process METHODOLOGY, PRODUCT, IRRELEVANT records without LLM validation
    Directly insert them into classified tables

    Args:
        cur: Database cursor
        conn: Database connection

    Returns:
        Dict with processing statistics
    """
    print(f"\n{'═' * 70}")
    print(f"📦 PROCESSING NON-VALIDATED CATEGORIES (METHODOLOGY, PRODUCT, IRRELEVANT)")
    print(f"{'═' * 70}")

    start_time = time.time()

    # Fetch records
    print(f"📥 Fetching records marked as METHODOLOGY, PRODUCT, or IRRELEVANT...")
    records = fetch_non_validated_other_categories(cur)

    if not records:
        print(f"   ⚠️  No records to process")
        return {
            'methodology': 0,
            'product': 0,
            'irrelevant': 0,
            'total': 0,
            'time': 0
        }

    print(f"   ✅ Found {len(records)} records to process")

    # Group by category
    category_counts = {'METHODOLOGY': 0, 'PRODUCT': 0, 'IRRELEVANT': 0}

    for record in records:
        category_counts[record['category']] += 1

    print(f"\n   📊 Breakdown:")
    print(f"      📚 METHODOLOGY: {category_counts['METHODOLOGY']}")
    print(f"      📦 PRODUCT: {category_counts['PRODUCT']}")
    print(f"      🗑️  IRRELEVANT: {category_counts['IRRELEVANT']}")

    # Process in batches
    print(f"\n   💾 Processing and inserting into classified tables...")
    batch_size = 50
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]

        for record in batch:
            try:
                # Insert into classified table
                handle_classified_insert(
                    cur,
                    record['category'],
                    record['extracted_name'],
                    record['description']
                )

                # Mark as processed in main table
                mark_as_processed(cur, record['extracted_name'], record['category'])

                total_inserted += 1

            except Exception as e:
                print(f"      ❌ Failed to process {record['extracted_name']}: {str(e)[:100]}")

        # Commit batch
        conn.commit()
        print(f"      ✅ Processed {min(i + batch_size, len(records))}/{len(records)} records")

    elapsed = time.time() - start_time

    print(f"\n{'═' * 70}")
    print(f"✅ NON-VALIDATED CATEGORIES COMPLETED")
    print(f"{'═' * 70}")
    print(f"   ⏱️  Time: {elapsed:.2f}s")
    print(f"   ✅ Total inserted: {total_inserted}")
    print(f"   📚 METHODOLOGY: {category_counts['METHODOLOGY']} → CLASSIFIED_METHODOLOGIES_DATA")
    print(f"   📦 PRODUCT: {category_counts['PRODUCT']} → CLASSIFIED_PRODUCTS_DATA")
    print(f"   🗑️  IRRELEVANT: {category_counts['IRRELEVANT']} → CLASSIFIED_IRRELEVANT_DATA")
    print(f"{'═' * 70}\n")

    return {
        'methodology': category_counts['METHODOLOGY'],
        'product': category_counts['PRODUCT'],
        'irrelevant': category_counts['IRRELEVANT'],
        'total': total_inserted,
        'time': elapsed
    }


# ============================================================================
# VALIDATION FUNCTIONS FOR EACH CATEGORY
# ============================================================================

def validate_tools_batch(records_batch):
    """
    Validate a batch of 3 TOOL records in one LLM call

    Args:
        records_batch: List of up to 3 records

    Returns:
        List of validation results matching input order
    """
    if not records_batch:
        return []

    # Build prompt for multiple items
    items_text = ""
    for idx, record in enumerate(records_batch, 1):
        items_text += f"""
**ITEM {idx}:**
- extracted_name: {record['extracted_name']}
- Description: {record['description']}
- Source: {record['source']}
- Current Reason: {record['reason']}
- Confidence Score: {record['confidence_score']}
"""

    prompt = f"""
You are an expert technical classification validator.
Your job is to validate whether each item is correctly classified as a TOOL.
====================
TOOL RULES
====================

R1 (Definition Rule):
A TOOL must be a specific software, platform, framework, simulator, IDE, automation system, testing tool, monitoring tool, modeling tool, or physical instrument/device used for:
design, development, analysis, testing, simulation, diagnostics, or automation.

R2 (Specificity Rule):
The item must be a clearly named, identifiable tool (e.g., Jenkins, MATLAB, Git).
Generic umbrella terms are NOT valid tools.

R3 (Industry Usage Rule):
The tool should be commonly recognized or used in industry/engineering/IT/automotive contexts.

R4 (No Generic Terms Rule):
Terms like “testing tools”, “software platforms”, “hardware”, “tools”, “systems” are INVALID.

R5 (Category Boundary Rule):
If the item is:
- a programming language or technical concept → TECHNOLOGY
- a standard/regulation → COMPLIANCE
- a process/methodology → INVALID TOOL

R6 (No Inference Rule):
Judge strictly based on the extracted_name itself — no assumptions.

====================
FEW-SHOT EXAMPLES
====================

VALID TOOL EXAMPLE:
[
  {{
    "item_number": 1,
    "extracted_name": "Jenkins",
    "is_valid": true,
    "reason": "Valid TOOL because Jenkins is a specific named automation server used for build and deployment processes, satisfying R1 as a tool for automation and R2 as a clearly identifiable industry-recognized tool."
  }}
]

INVALID TOOL (GENERIC):
[
  {{
    "item_number": 1,
    "extracted_name": "testing tools",
    "is_valid": false,
    "reason": "Invalid TOOL because 'testing tools' is a generic umbrella term rather than a specific named tool, violating R2 (specificity rule) and R4 (no generic terms rule)."
  }}
]
**Examples of TOOLS:**
- Jenkins (CI/CD tool)
- JIRA (Project management tool)
- Git (Version control tool)
- Docker (Containerization tool)
- Selenium (Testing tool)
- Visual Studio Code (IDE/Editor tool)

**NOT TOOLS:**
- Programming languages (e.g., Python, Java) → These are TECHNOLOGIES
- Standards or regulations (e.g., ISO 27001, GDPR) → These are COMPLIANCES
- Methodologies (e.g., Agile, Scrum) → These are METHODOLOGIES
- Products sold as solutions (e.g., iPhone, AWS Cloud) → These are PRODUCTS

**Items to Validate:**
{items_text}

====================
IMPORTANT FOR REASON FIELD
====================

Each reason MUST:
- Explain clearly  WHY valid or invalid
- Explicitly reference rule numbers (R1, R2, etc.)
- Link explanation to TOOL definition

**Response Format (ONLY valid JSON array):**
[
    {{
        "item_number": 1,
        "extracted_name": "exact extracted_name from item 1",
        "is_valid": true or false,
        "reason": "Explanation + rule references mentioning why it is valid or invalid"
    }},
    {{
        "item_number": 2,
        "extracted_name": "exact extracted_name from item 2",
        "is_valid": true or false,
        "reason": "..."
    }},
    ...
]

**CRITICAL:**
- Respond with a JSON array with exactly {len(records_batch)} objects
- Each object must have: item_number, extracted_name, is_valid, reason
- Be specific and reference the actual characteristics
- Respond ONLY with valid JSON, no additional text
"""

    response_text, api_time = call_deepseek(prompt)

    if not response_text:
        return None, api_time

    try:
        # Clean response
        response_clean = response_text.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        results = json.loads(response_clean)

        # Validate we got the right number of results
        if not isinstance(results, list) or len(results) != len(records_batch):
            print(
                f"      ⚠️  Expected {len(records_batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}")
            return None, api_time

        return results, api_time

    except json.JSONDecodeError as e:
        print(f"      ❌ JSON Parse Error: {str(e)[:100]}")
        return None, api_time


def validate_technologies_batch(records_batch):
    """
    Validate a batch of 3 TECHNOLOGY records in one LLM call

    Args:
        records_batch: List of up to 3 records

    Returns:
        List of validation results matching input order
    """
    if not records_batch:
        return []

    # Build prompt for multiple items
    items_text = ""
    for idx, record in enumerate(records_batch, 1):
        items_text += f"""
**ITEM {idx}:**
- Extracted_name: {record['extracted_name']}
- Description: {record['description']}
- Source: {record['source']}
- Current Reason: {record['reason']}
- Confidence Score: {record['confidence_score']}
"""

    prompt = f"""
You are an expert validator for technical classification.

Validate whether each item is correctly classified as a TECHNOLOGY. 
====================
TECHNOLOGY RULES
====================

R1 (Definition Rule):
A TECHNOLOGY is a technical concept, programming language, system architecture, algorithm, protocol, innovation, computing paradigm, or foundational technical method/approach. Technologies represent the underlying concepts, languages, and systems rather than specific implementations or tools.

R2 (Specific Named Rule):
Must be a clearly named technology (e.g., Python, Linux, REST API, Machine Learning).

R3 (Technical Nature Rule):
Must represent how systems are built or operate, not a tool that performs tasks.

R4 (No Generic Concepts Rule):
Generic terms like “programming concepts”, “algorithms”, “software knowledge” are INVALID or very weak.

R5 (Category Boundary Rule):
If the item is:
- a specific software/tool → TOOL
- a regulation/standard → COMPLIANCE

R6 (No Inference Rule):
Judge only by the item name.

====================
FEW-SHOT EXAMPLES — TECHNOLOGY
====================

VALID TECHNOLOGY EXAMPLE:
[
  {{
    "item_number": 1,
    "extracted_name": "Python",
    "is_valid": true,
    "reason": "Valid TECHNOLOGY because Python is a clearly named programming language representing a foundational technical method for building software systems, satisfying R1 as a technical concept and R2 as a specific named technology."
  }}
]

INVALID TECHNOLOGY — GENERIC TERM:
[
  {{
    "item_number": 1,
    "extracted_name": "programming concepts",
    "is_valid": false,
    "reason": "Invalid TECHNOLOGY because 'programming concepts' is a generic umbrella term rather than a clearly named specific technology, violating R2 (specific named rule) and R4 (no generic concepts rule)."
  }}
]

**Examples of TECHNOLOGIES:**
- Python (Programming language)
- Machine Learning (Technical concept/approach)
- Blockchain (Technology/system)
- REST API (Protocol/architectural style)
- Cloud Computing (Computing paradigm)
- Artificial Intelligence (Technical field/approach)
- Microservices (Architecture pattern)

**NOT TECHNOLOGIES:**
- Specific tools or applications (e.g., Jenkins, JIRA) → These are TOOLS
- Standards or regulations (e.g., PCI-DSS, HIPAA) → These are COMPLIANCES
- Processes or practices (e.g., Agile, DevOps) → These are METHODOLOGIES
- Commercial products (e.g., Salesforce, Oracle DB) → These are PRODUCTS


====================
REASON FORMAT RULE
====================

Each "reason" MUST:

- Clearly explain WHY the item is valid or invalid in natural language
- Explicitly reference applicable rule numbers (R1, R2, R3, etc.)
- Link explanation directly to the TECHNOLOGY definition

**Items to Validate:**
{items_text}

**Response Format (ONLY valid JSON array):**
[
    {{
        "item_number": 1,
        "extracted_name": "exact extracted_name from item 1",
        "is_valid": true or false,
        "reason": "Valid/Invald TECHNOLOGY because <clear explanation>, satisfying/Violating R# (...) and R# (...)."
    }},
    {{
        "item_number": 2,
        "extracted_name": "exact extracted_name from item 2",
        "is_valid": true or false,
        "reason": "Valid/Invald TECHNOLOGY because <clear explanation>, satisfying/Violating R# (...) and R# (...)"
    }},
    ...
]

**CRITICAL:**
- Respond with a JSON array with exactly {len(records_batch)} objects
- Each object must have: item_number, extracted_name, is_valid, reason
- Be specific and reference the actual characteristics
- Respond ONLY with valid JSON, no additional text
"""

    response_text, api_time = call_deepseek(prompt)

    if not response_text:
        return None, api_time

    try:
        # Clean response
        response_clean = response_text.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        results = json.loads(response_clean)

        # Validate we got the right number of results
        if not isinstance(results, list) or len(results) != len(records_batch):
            print(
                f"      ⚠️  Expected {len(records_batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}")
            return None, api_time

        return results, api_time

    except json.JSONDecodeError as e:
        print(f"      ❌ JSON Parse Error: {str(e)[:100]}")
        return None, api_time


def validate_compliances_batch(records_batch):
    """
    Validate a batch of 3 COMPLIANCE records in one LLM call

    Args:
        records_batch: List of up to 3 records

    Returns:
        List of validation results matching input order
    """
    if not records_batch:
        return []

    # Build prompt for multiple items
    items_text = ""
    for idx, record in enumerate(records_batch, 1):
        items_text += f"""
**ITEM {idx}:**
- Extracted_name: {record['extracted_name']}
- Description: {record['description']}
- Source: {record['source']}
- Current Reason: {record['reason']}
- Confidence Score: {record['confidence_score']}
"""

    prompt = f"""
You are an expert validator for technical classification. 
Validate whether each item is correctly classified as a COMPLIANCE.

====================
COMPLIANCE RULES
====================

R1 (Definition Rule):
A COMPLIANCE is a regulation, law, standard, certification, policy, requirement, or guideline mandated by regulatory bodies, government authorities, industry organizations, or international standards bodies. Compliances are requirements that organizations must adhere to.
R2 (Named Standard Rule):
Must be a clearly named, recognized compliance (e.g., ISO 26262, GDPR, IATF 16949).

R3 (Regulatory Nature Rule):
Must impose rules, requirements, or obligations.

R4 (No Generic References Rule):
Generic terms like “company standards”, “rules”, “safety guidelines” are INVALID.

R5 (Category Boundary Rule):
If the item is:
- software/tool → TOOL
- technical method/concept → TECHNOLOGY

R6 (No Inference Rule):
Judge strictly by the term itself.

====================
FEW-SHOT EXAMPLES — COMPLIANCE
====================

VALID COMPLIANCE EXAMPLE:
[
  {{
    "item_number": 1,
    "extracted_name": "ISO 26262",
    "is_valid": true,
    "reason": "Valid COMPLIANCE because ISO 26262 is a clearly named functional safety standard issued by an international standards body defining mandatory safety requirements, satisfying R1 as a regulatory standard and R2 as a recognized named compliance."
  }}
]

INVALID COMPLIANCE — GENERIC TERM:
[
  {{
    "item_number": 1,
    "extracted_name": "company standards",
    "is_valid": false,
    "reason": "Invalid COMPLIANCE because 'company standards' is a generic reference without naming a specific recognized regulation or standard, violating R2 (named standard rule) and R4 (no generic references rule)."
  }}
]
**Examples of COMPLIANCES:**
- GDPR (General Data Protection Regulation)
- ISO 27001 (Information security standard)
- HIPAA (Health Insurance Portability and Accountability Act)
- PCI-DSS (Payment Card Industry Data Security Standard)
- SOX (Sarbanes-Oxley Act)
- FDA regulations
- SOC 2 (Service Organization Control 2)

**NOT COMPLIANCES:**
- Tools for compliance management (e.g., compliance software) → These are TOOLS
- Technologies used for compliance (e.g., encryption) → These are TECHNOLOGIES
- Best practices or methodologies (e.g., Agile, ITIL) → These are METHODOLOGIES
- Products that help with compliance → These are PRODUCTS

====================
REASON FORMAT RULE
====================

Each "reason" MUST:

- Clearly explain WHY the item is valid or invalid in natural language
- Explicitly reference applicable rule numbers (R1, R2, R3, etc.)
- Link explanation directly to the COMPLIANCE definition

**Items to Validate:**
{items_text}

**Response Format (ONLY valid JSON array):**
[
    {{
        "item_number": 1,
        "extracted_name": "exact extracted_name from item 1",
        "is_valid": true or false,
        "reason": "Valid/Invalid COMPLIANCE because <clear explanation>, satisfying/Violating R# (...) and R# (...)."
    }},
    {{
        "item_number": 2,
        "extracted_name": "exact extracted_name from item 2",
        "is_valid": true or false,
         "reason": "Valid/Invalid COMPLIANCE because <clear explanation>, satisfying/Violating R# (...) and R# (...)."

    }},
    ...
]

**CRITICAL:**
- Respond with a JSON array with exactly {len(records_batch)} objects
- Each object must have: item_number, extracted_name, is_valid, reason
- Be specific and reference the actual characteristics
- Respond ONLY with valid JSON, no additional text
"""

    response_text, api_time = call_deepseek(prompt)

    if not response_text:
        return None, api_time

    try:
        # Clean response
        response_clean = response_text.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()

        results = json.loads(response_clean)

        # Validate we got the right number of results
        if not isinstance(results, list) or len(results) != len(records_batch):
            print(
                f"      ⚠️  Expected {len(records_batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}")
            return None, api_time

        return results, api_time

    except json.JSONDecodeError as e:
        print(f"      ❌ JSON Parse Error: {str(e)[:100]}")
        return None, api_time


# ============================================================================
# DATABASE UPDATE
# ============================================================================

def update_validation_results(cur, validation_results, category):
    """
    Update multiple records with validation results

    Args:
        cur: Database cursor
        validation_results: List of dicts with extracted_name, is_valid, reason
        category: The category being validated (TOOL/TECHNOLOGY/COMPLIANCE)

    Returns:
        Count of successfully updated records
    """
    updated_count = 0

    for result in validation_results:
        valid_flag = 'VALID' if result['is_valid'] else 'INVALID'

        validation_json = {
            "valid_flag": valid_flag,
            "reason": result['reason'],
            "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": category
        }

        # For VALID records: Update UPDATED_CATEGORY, VALIDATED_BY, and COMMENTS
        # For INVALID records: Only update IS_VALID and VALIDATION_RESULT (for manual review)
        if result['is_valid']:
            update_query = """
            UPDATE "PROCESSING"."MASTERDATA_LLM_CLASSIFICATION_VALIDATION"
            SET 
                "IS_VALID" = %s,
                "UPDATED_TIMESTAMP" = %s,
                "VALIDATION_RESULT" = %s,
                "UPDATED_CATEGORY" = %s,
                "VALIDATED_BY" = %s,
                "COMMENTS" = %s
            WHERE "EXTRACTED_NAME" = %s
            """

            try:
                cur.execute(
                    update_query,
                    (
                        valid_flag,
                        datetime.now(),
                        Json(validation_json),
                        category,  # UPDATED_CATEGORY
                        'DEEPSEEK',  # VALIDATED_BY
                        result['reason'],  # COMMENTS (validation reason)
                        result['extracted_name']
                    )
                )
                updated_count += 1
            except Exception as e:
                print(f"      ❌ Failed to update {result['extracted_name']}: {str(e)[:100]}")
        else:
            # INVALID records - only update IS_VALID and VALIDATION_RESULT (manual review needed)
            update_query = """
            UPDATE "PROCESSING"."MASTERDATA_LLM_CLASSIFICATION_VALIDATION"
            SET 
                "IS_VALID" = %s,
                "UPDATED_TIMESTAMP" = %s,
                "VALIDATION_RESULT" = %s
            WHERE "EXTRACTED_NAME" = %s
            """

            try:
                cur.execute(
                    update_query,
                    (
                        valid_flag,
                        datetime.now(),
                        Json(validation_json),
                        result['extracted_name']
                    )
                )
                updated_count += 1
            except Exception as e:
                print(f"      ❌ Failed to update {result['extracted_name']}: {str(e)[:100]}")

    return updated_count


# ============================================================================
# PROCESS CATEGORY WITH OPTIMIZED BATCHING
# ============================================================================

def process_category(cur, conn, category):
    print(f"\n{'═' * 70}")
    print(f"{'🔧' if category == 'TOOL' else '💻' if category == 'TECHNOLOGY' else '📋'} PROCESSING CATEGORY: {category}")
    print(f"{'═' * 70}")

    category_start_time = time.time()

    if category == 'TOOL':
        validate_func = validate_tools_batch
    elif category == 'TECHNOLOGY':
        validate_func = validate_technologies_batch
    elif category == 'COMPLIANCE':
        validate_func = validate_compliances_batch
    else:
        print(f"❌ Unknown category: {category}")
        return {'valid': 0, 'invalid': 0, 'failed': 0, 'total_time': 0, 'total_records': 0, 'api_calls': 0, 'api_time': 0}

    print(f"📥 Fetching {category} records from database...")
    fetch_start = time.time()
    all_records = fetch_category_records(cur, category)
    fetch_time = time.time() - fetch_start

    total_records = len(all_records)

    if total_records == 0:
        print(f"   ⚠️  No {category} records to process")
        return {'valid': 0, 'invalid': 0, 'failed': 0, 'total_time': 0, 'total_records': 0, 'api_calls': 0, 'api_time': 0}

    print(f"   ✅ Fetched {total_records} records in {fetch_time:.2f}s")

    num_batches = (total_records + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   📦 Will process in {num_batches} batches of up to {BATCH_SIZE} records")
    print(f"   🔄 Each batch will make LLM calls with {LLM_BATCH_SIZE} items at a time IN PARALLEL")

    total_valid = 0
    total_invalid = 0
    total_failed = 0
    total_api_calls = 0
    total_api_time = 0

    for batch_idx in range(0, total_records, BATCH_SIZE):
        batch_num = (batch_idx // BATCH_SIZE) + 1
        batch_end = min(batch_idx + BATCH_SIZE, total_records)
        batch_records = all_records[batch_idx:batch_end]
        batch_size = len(batch_records)

        print(f"\n{'─' * 70}")
        print(f"📦 BATCH {batch_num}/{num_batches} ({batch_size} records)")
        print(f"{'─' * 70}")

        batch_start_time = time.time()
        batch_validation_results = []
        batch_valid = 0
        batch_invalid = 0
        batch_failed = 0
        batch_api_calls = 0
        batch_api_time = 0

        # ── BUILD ALL LLM SUB-BATCHES ──────────────────────────────────────
        llm_sub_batches = []
        for llm_batch_idx in range(0, batch_size, LLM_BATCH_SIZE):
            llm_batch_end = min(llm_batch_idx + LLM_BATCH_SIZE, batch_size)
            llm_sub_batches.append(batch_records[llm_batch_idx:llm_batch_end])

        num_llm_calls = len(llm_sub_batches)
        print(f"   ⚡ Firing {num_llm_calls} LLM calls IN PARALLEL...")

        # ── FIRE ALL LLM CALLS IN PARALLEL ────────────────────────────────
        async def run_llm_calls_async(sub_batches, validate_fn):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=len(sub_batches)) as executor:
                tasks = [
                    loop.run_in_executor(executor, validate_fn, sub_batch)
                    for sub_batch in sub_batches
                ]
                return await asyncio.gather(*tasks)

        llm_start = time.time()
        all_llm_results = asyncio.run(run_llm_calls_async(llm_sub_batches, validate_func))
        llm_total_elapsed = time.time() - llm_start
        print(f"   ⚡ All {num_llm_calls} parallel LLM calls completed in {llm_total_elapsed:.2f}s")

        # ── PROCESS RESULTS FROM ALL PARALLEL CALLS ───────────────────────
        for llm_call_num, (llm_batch_records, (results, api_time)) in enumerate(
            zip(llm_sub_batches, all_llm_results), 1
        ):
            llm_batch_count = len(llm_batch_records)
            batch_api_calls += 1
            batch_api_time += api_time if api_time else 0

            if results is None:
                print(f"      ❌ LLM Call {llm_call_num}/{num_llm_calls} Failed")
                batch_failed += llm_batch_count
                continue

            if len(results) != llm_batch_count:
                print(f"      ⚠️  LLM Call {llm_call_num}/{num_llm_calls} Mismatch: expected {llm_batch_count}, got {len(results)}")
                batch_failed += llm_batch_count
                continue

            print(f"      ✅ LLM Call {llm_call_num}/{num_llm_calls} ({llm_batch_count} items, API: {api_time:.2f}s)")

            for idx, result in enumerate(results):
                expected_extracted_name = llm_batch_records[idx]['extracted_name']
                if result.get('extracted_name') != expected_extracted_name:
                    print(f"         ⚠️  extracted_name mismatch: expected '{expected_extracted_name}', got '{result.get('extracted_name')}'")
                    batch_failed += 1
                    continue

                batch_validation_results.append(result)

                if result['is_valid']:
                    batch_valid += 1
                    status = "✅ VALID"
                else:
                    batch_invalid += 1
                    status = "❌ INVALID (Manual Review)"

                print(f"         {idx + 1}. {result['extracted_name'][:30]:30s} → {status}")

        # ── DB COMMIT PER BATCH ────────────────────────────────────────────
        print(f"\n   💾 Updating database for batch {batch_num}...", end=" ")
        db_update_start = time.time()
        updated = update_validation_results(cur, batch_validation_results, category)
        conn.commit()
        db_update_time = time.time() - db_update_start
        print(f"✅ {updated} records updated ({db_update_time:.2f}s)")

        valid_in_batch = sum(1 for r in batch_validation_results if r['is_valid'])
        invalid_in_batch = len(batch_validation_results) - valid_in_batch
        print(f"      ✅ {valid_in_batch} marked VALID (UPDATED_CATEGORY={category}, VALIDATED_BY=DEEPSEEK)")
        print(f"      ❌ {invalid_in_batch} marked INVALID (Flagged for manual review)")

        batch_elapsed = time.time() - batch_start_time

        print(f"\n   📊 BATCH {batch_num} SUMMARY:")
        print(f"      ⏱️  Total time: {batch_elapsed:.2f}s")
        print(f"      🔄 API calls: {batch_api_calls} (total API time: {batch_api_time:.2f}s)")
        print(f"      ✅ Valid: {batch_valid}")
        print(f"      ❌ Invalid: {batch_invalid}")
        print(f"      ⚠️  Failed: {batch_failed}")
        print(f"      ⚡ Avg per record: {batch_elapsed / batch_size:.2f}s")

        total_valid += batch_valid
        total_invalid += batch_invalid
        total_failed += batch_failed
        total_api_calls += batch_api_calls
        total_api_time += batch_api_time

        progress = (batch_end / total_records) * 100
        print(f"   📈 Overall progress: {batch_end}/{total_records} ({progress:.1f}%)")

    category_elapsed = time.time() - category_start_time

    print(f"\n{'═' * 70}")
    print(f"✅ {category} CATEGORY COMPLETED")
    print(f"{'═' * 70}")
    print(f"⏱️  TIMING:")
    print(f"   Total time: {category_elapsed:.2f}s ({category_elapsed / 60:.2f} min)")
    print(f"   Data fetch: {fetch_time:.2f}s")
    print(f"   API calls: {total_api_calls} (total API time: {total_api_time:.2f}s)")
    print(f"   Avg API time per call: {total_api_time / total_api_calls:.2f}s" if total_api_calls > 0 else "")
    print(f"   Avg total time per record: {category_elapsed / total_records:.2f}s")
    print(f"\n📊 RESULTS:")
    print(f"   Total records: {total_records}")
    print(f"   ✅ Valid: {total_valid} ({total_valid / total_records * 100:.1f}%) - Auto-approved")
    print(f"   ❌ Invalid: {total_invalid} ({total_invalid / total_records * 100:.1f}%) - Manual review required")
    print(f"   ⚠️  Failed: {total_failed} ({total_failed / total_records * 100:.1f}%)")
    if total_records > 0:
        success_rate = (total_valid + total_invalid) / total_records * 100
        print(f"   📈 Success rate: {success_rate:.1f}%")
    print(f"{'═' * 70}\n")

    return {
        'valid': total_valid,
        'invalid': total_invalid,
        'failed': total_failed,
        'total_time': category_elapsed,
        'total_records': total_records,
        'api_calls': total_api_calls,
        'api_time': total_api_time
    }

# ============================================================================
# LOGGING
# ============================================================================

def insert_log(cur, category_stats, other_stats):
    """Insert log entry into TTC_MASTER_DATA_LOGS"""
    print(f"\n📝 Inserting log into TTC_MASTER_DATA_LOGS...")

    cur.execute("""
        SELECT COALESCE(MAX(CAST(SUBSTRING("LOG_ID" FROM 'LOG_(\\d+)') AS INTEGER)), 0) + 1
        FROM "LOGS"."PROCESSED_MASTER_DATA_LOGS"
    """)
    next_id = cur.fetchone()[0]
    log_id = f"LOG_{next_id:07d}"

    # Safely calculate totals with default values
    total_validated = sum(stats.get('total_records', 0) for stats in category_stats.values())
    total_valid = sum(stats.get('valid', 0) for stats in category_stats.values())
    total_other = other_stats.get('total', 0)

    total_processed = total_validated + total_other

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
            "LLM_VALIDATION_WITH_AUTO_CLASSIFICATION",
            "MASTERDATA_LLM_CLASSIFICATION_VALIDATION",
            total_processed,
            total_valid + total_other,
            total_processed,
            datetime.now()
        )
    )

    print(f"   ✅ Log inserted: {log_id}")
    return log_id


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def validate_llm_classifications():
    """Main function: Optimized validation with batching"""

    print("\n" + "=" * 70)
    print("🚀 OPTIMIZED LLM CLASSIFICATION VALIDATION")
    print("=" * 70)
    print(f"📋 Configuration:")
    print(f"   Model: {DEEPSEEK_MODEL}")
    print(f"   Batch size (DB commit): {BATCH_SIZE} records")
    print(f"   LLM batch size: {LLM_BATCH_SIZE} items per API call")
    print(f"\n📝 Processing Strategy:")
    print(f"   🔍 LLM VALIDATION: TOOL, TECHNOLOGY, COMPLIANCE")
    print(f"   ⚡ AUTO-CLASSIFICATION (No LLM): METHODOLOGY, PRODUCT, IRRELEVANT")
    print(f"\n📝 Update Strategy:")
    print(f"   ✅ VALID (LLM) → UPDATED_CATEGORY, VALIDATED_BY=DEEPSEEK, COMMENTS")
    print(f"   ❌ INVALID (LLM) → Flagged for manual review")
    print(f"   ⚡ AUTO → Insert into CLASSIFIED_*_DATA tables")
    print("=" * 70)

    conn, cur = connect_to_db()

    try:
        overall_start = time.time()

        # First, process other categories (no LLM validation)
        other_stats = process_other_categories(cur, conn)

        # Then, process all 3 LLM validation categories IN PARALLEL
        # Each category gets its own isolated DB connection
        def run_category(category):
            cat_conn, cat_cur = connect_to_db()
            try:
                result = process_category(cat_cur, cat_conn, category)
                return category, result
            except Exception as e:
                print(f"\n❌ [{category}] Failed: {str(e)}")
                return category, {
                    'valid': 0, 'invalid': 0, 'failed': 0,
                    'total_time': 0, 'total_records': 0,
                    'api_calls': 0, 'api_time': 0
                }
            finally:
                cat_cur.close()
                cat_conn.close()

        print(f"\n{'█' * 70}")
        print(f"⚡ RUNNING TOOL, TECHNOLOGY, COMPLIANCE VALIDATION IN PARALLEL")
        print(f"{'█' * 70}")

        category_stats = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_category, category): category
                for category in ['TOOL', 'TECHNOLOGY', 'COMPLIANCE']
            }
            for future in as_completed(futures):
                category, result = future.result()
                category_stats[category] = result

        # Insert log
        log_id = insert_log(cur, category_stats, other_stats)
        conn.commit()

        overall_elapsed = time.time() - overall_start

        # Final summary with safe handling of missing keys
        total_validated = sum(stats.get('total_records', 0) for stats in category_stats.values())
        total_valid = sum(stats.get('valid', 0) for stats in category_stats.values())
        total_invalid = sum(stats.get('invalid', 0) for stats in category_stats.values())
        total_failed = sum(stats.get('failed', 0) for stats in category_stats.values())
        total_api_calls = sum(stats.get('api_calls', 0) for stats in category_stats.values())
        total_api_time = sum(stats.get('api_time', 0) for stats in category_stats.values())

        total_auto_classified = other_stats.get('total', 0)
        total_all = total_validated + total_auto_classified

        print("\n" + "=" * 70)
        print("✅ ALL PROCESSING COMPLETED")
        print("=" * 70)

        if total_all == 0:
            print("\n⚠️  No records were processed - all categories are already validated!")
            print(f"   📝 Log ID: {log_id}")
            print("=" * 70)
            return

        print(f"\n⏱️  OVERALL TIMING:")
        print(f"   Total execution time: {overall_elapsed:.2f}s ({overall_elapsed / 60:.2f} min)")
        print(f"   Total API calls: {total_api_calls}")
        print(f"   Total API time: {total_api_time:.2f}s")
        if total_api_calls > 0:
            print(f"   Avg API time per call: {total_api_time / total_api_calls:.2f}s")

        print(f"\n📊 LLM VALIDATED CATEGORIES (TOOL, TECHNOLOGY, COMPLIANCE):")
        print(f"   {'─' * 66}")
        for category, stats in category_stats.items():
            if stats.get('total_records', 0) > 0:
                total_cat_records = stats['total_records']
                print(f"   {category}:")
                print(f"      Records: {total_cat_records}")
                print(f"      Time: {stats['total_time']:.2f}s ({stats['total_time'] / 60:.2f} min)")
                print(f"      ✅ Valid: {stats['valid']} ({stats['valid'] / total_cat_records * 100:.1f}%)")
                print(f"      ❌ Invalid: {stats['invalid']} ({stats['invalid'] / total_cat_records * 100:.1f}%)")
                print(f"      ⚠️  Failed: {stats['failed']}")
                print(f"      API calls: {stats['api_calls']}")
                print(f"      Avg per record: {stats['total_time'] / total_cat_records:.2f}s")
                print(f"   {'─' * 66}")

        print(f"\n📦 AUTO-CLASSIFIED CATEGORIES (No LLM):")
        print(f"   {'─' * 66}")
        print(f"   Total processed: {total_auto_classified}")
        print(f"   Time: {other_stats.get('time', 0):.2f}s")
        print(f"   📚 METHODOLOGY: {other_stats.get('methodology', 0)}")
        print(f"   📦 PRODUCT: {other_stats.get('product', 0)}")
        print(f"   🗑️  IRRELEVANT: {other_stats.get('irrelevant', 0)}")
        print(f"   {'─' * 66}")

        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total records processed: {total_all}")
        print(f"   ├─ LLM Validated: {total_validated}")
        if total_validated > 0:
            print(f"   │  ├─ ✅ Valid: {total_valid} ({total_valid / total_validated * 100:.1f}%)")
            print(f"   │  ├─ ❌ Invalid: {total_invalid} ({total_invalid / total_validated * 100:.1f}%)")
            print(f"   │  └─ ⚠️  Failed: {total_failed}")
        print(f"   └─ ⚡ Auto-Classified: {total_auto_classified}")

        print(f"\n📝 Database Updates:")
        print(f"   ✅ {total_valid} LLM validated records: UPDATED_CATEGORY set, VALIDATED_BY=DEEPSEEK")
        print(f"   ⚡ {total_auto_classified} auto-classified records: Inserted into CLASSIFIED_*_DATA tables")
        print(f"   ❌ {total_invalid} records: Flagged as INVALID for manual review")

        print(f"\n   📝 Log ID: {log_id}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        conn.rollback()

    finally:
        cur.close()
        conn.close()
        print("\n✅ Database connection closed")


# if __name__ == "__main__":
#     print("\n" + "🚀" * 35)
#     print("   OPTIMIZED BATCH LLM VALIDATION WITH AUTO-CLASSIFICATION")
#     print("🚀" * 35 + "\n")
#     validate_llm_classifications()