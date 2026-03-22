import psycopg2
from datetime import datetime
import sys
from collections import defaultdict
import math

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

# ============================================================================
# SET CATEGORY HERE - CHANGE THIS TO RUN DIFFERENT PIPELINES
# ============================================================================
CATEGORY = "COMPLIANCES"  # Options: "TOOLS", "TECHNOLOGIES", "COMPLIANCES"

# Get configuration for selected category
CONFIG = CATEGORY_CONFIG[CATEGORY]

SCHEMA = "PROCESSING"

# Validator Configuration
VALIDATORS = {
    "validator_1": {
        "name": "Madhu Mitha",
        "is_available": True  # Can be set to False to skip this validator
    },
    "validator_2": {
        "name": "Shemanithi",
        "is_available": True
    },
    "validator_3": {
        "name": "Gopikrishna A.V",
        "is_available": True
    }
}


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
# DATA FETCHING FUNCTIONS
# ============================================================================

def get_assignment_statistics(cur):
    """Get current assignment statistics"""
    print("\n📊 Fetching current assignment statistics...")

    query = """
    SELECT 
        "ASSIGNED_VALIDATOR",
        "IS_VALID",
        COUNT(*) as count
    FROM "{SCHEMA}"."{PIPELINE_TABLES['LLM_CLASSIFICATION']}"
    GROUP BY "ASSIGNED_VALIDATOR", "IS_VALID"
    ORDER BY "ASSIGNED_VALIDATOR" NULLS FIRST, "IS_VALID"
    """

    cur.execute(query)
    rows = cur.fetchall()

    stats = defaultdict(lambda: {'VALID': 0, 'INVALID': 0, 'OTHER': 0})

    for row in rows:
        validator = row[0] if row[0] else "UNASSIGNED"
        is_valid = row[1] if row[1] else "OTHER"
        count = row[2]

        if is_valid in ['VALID', 'INVALID']:
            stats[validator][is_valid] = count
        else:
            stats[validator]['OTHER'] = count

    print(f"\n   Current Assignment Status:")
    for validator in sorted(stats.keys()):
        valid_count = stats[validator]['VALID']
        invalid_count = stats[validator]['INVALID']
        other_count = stats[validator]['OTHER']
        total = valid_count + invalid_count + other_count
        print(
            f"      {validator}: {total} records (VALID: {valid_count}, INVALID: {invalid_count}, OTHER: {other_count})")

    return stats


def fetch_unassigned_records_by_validity(cur):
    """
    Fetch unassigned records separately for VALID and INVALID categories
    Returns a dictionary with 'VALID' and 'INVALID' keys
    """
    print("\n🔍 Fetching unassigned records from LLM_CLASSIFICATION...")

    # Fetch VALID records
    query_valid = """
    SELECT "WORD"
    FROM "{SCHEMA}"."{PIPELINE_TABLES['LLM_CLASSIFICATION']}"
    WHERE "ASSIGNED_VALIDATOR" IS NULL 
      AND "IS_VALID" = 'VALID'
    ORDER BY "INSERTED_TIMESTAMP" DESC
    """

    cur.execute(query_valid)
    valid_rows = cur.fetchall()

    valid_records = [{'word': row[0], 'is_valid': 'VALID'} for row in valid_rows]

    print(f"   ✅ Found {len(valid_records)} VALID unassigned records")

    # Fetch INVALID records
    query_invalid = """
    SELECT "WORD"
    FROM "{SCHEMA}"."{PIPELINE_TABLES['LLM_CLASSIFICATION']}"
    WHERE "ASSIGNED_VALIDATOR" IS NULL 
      AND "IS_VALID" = 'INVALID'
    ORDER BY "INSERTED_TIMESTAMP" DESC
    """

    cur.execute(query_invalid)
    invalid_rows = cur.fetchall()

    invalid_records = [{'word': row[0], 'is_valid': 'INVALID'} for row in invalid_rows]

    print(f"   ✅ Found {len(invalid_records)} INVALID unassigned records")

    return {
        'VALID': valid_records,
        'INVALID': invalid_records
    }


# ============================================================================
# VALIDATOR SELECTION
# ============================================================================

def get_available_validators():
    """Get list of available validators"""
    available = []
    for val_key, val_data in VALIDATORS.items():
        if val_data['is_available']:
            available.append(val_data['name'])

    print(f"\n👥 Available Validators: {len(available)}")
    for idx, name in enumerate(available, 1):
        print(f"   {idx}. {name}")

    return available


def distribute_records(records, validators, category):
    """
    Distribute records equally among available validators
    Handles remainders by assigning to first N validators

    Args:
        records: List of record dicts to assign
        validators: List of validator names
        category: 'VALID' or 'INVALID' for logging

    Returns:
        Dict mapping validator names to lists of records
    """
    print(f"\n📦 Distributing {len(records)} {category} records among {len(validators)} validators...")

    if not validators:
        print("   ⚠️  No validators available!")
        return {}

    if not records:
        print(f"   ⚠️  No {category} records to distribute!")
        return {}

    # Calculate distribution
    total_records = len(records)
    num_validators = len(validators)

    base_count = total_records // num_validators
    remainder = total_records % num_validators

    print(f"\n   Distribution Plan for {category}:")
    print(f"      Base allocation per validator: {base_count}")
    print(f"      Remainder: {remainder}")

    # Create assignment dictionary
    assignments = {validator: [] for validator in validators}

    # Distribute records
    record_idx = 0
    for validator_idx, validator in enumerate(validators):
        # Calculate how many records this validator gets
        count = base_count
        if validator_idx < remainder:
            count += 1  # Give extra record to first N validators

        # Assign records
        for _ in range(count):
            if record_idx < total_records:
                assignments[validator].append(records[record_idx])
                record_idx += 1

        print(f"      {validator}: {count} {category} records")

    print(f"\n   ✅ {category} distribution complete: {record_idx}/{total_records} records assigned")

    return assignments


def merge_assignments(valid_assignments, invalid_assignments):
    """
    Merge VALID and INVALID assignments for each validator

    Args:
        valid_assignments: Dict of validator -> VALID records
        invalid_assignments: Dict of validator -> INVALID records

    Returns:
        Combined assignments dict with all records per validator
    """
    print(f"\n🔀 Merging VALID and INVALID assignments...")

    combined = defaultdict(list)

    # Add VALID records
    for validator, records in valid_assignments.items():
        combined[validator].extend(records)

    # Add INVALID records
    for validator, records in invalid_assignments.items():
        combined[validator].extend(records)

    # Print summary
    print(f"\n   Combined Assignment Summary:")
    for validator in sorted(combined.keys()):
        valid_count = sum(1 for r in combined[validator] if r.get('is_valid') == 'VALID')
        invalid_count = sum(1 for r in combined[validator] if r.get('is_valid') == 'INVALID')
        total = len(combined[validator])
        print(f"      {validator}: {total} total (VALID: {valid_count}, INVALID: {invalid_count})")

    return dict(combined)


# ============================================================================
# DATABASE UPDATE FUNCTIONS
# ============================================================================

def assign_validators_to_records(cur, conn, assignments, batch_size=30):
    """
    Batch update validator assignments with progress logging
    Tracks VALID and INVALID separately
    (UPDATED_TIMESTAMP is NOT modified)
    """

    print(f"\n💾 Updating database in batches of {batch_size}...")

    update_query = """
    UPDATE "{SCHEMA}"."{PIPELINE_TABLES['LLM_CLASSIFICATION']}"
    SET "ASSIGNED_VALIDATOR" = %s
    WHERE "WORD" = ANY(%s)
    """

    stats = defaultdict(lambda: {
        'assigned': 0,
        'failed': 0,
        'total': 0,
        'valid_assigned': 0,
        'invalid_assigned': 0
    })
    total_updated = 0

    for validator, records in assignments.items():
        print(f"\n   📝 Assigning to {validator} in batches...")

        words = [r['word'] for r in records]

        # Count VALID and INVALID for this validator
        valid_count = sum(1 for r in records if r.get('is_valid') == 'VALID')
        invalid_count = sum(1 for r in records if r.get('is_valid') == 'INVALID')

        successful = 0
        failed = 0

        total_batches = math.ceil(len(words) / batch_size)

        print(f"      📦 Total batches for {validator}: {total_batches}")
        print(f"      📊 Records: {len(records)} total (VALID: {valid_count}, INVALID: {invalid_count})")

        for batch_num, i in enumerate(range(0, len(words), batch_size), start=1):
            batch_words = words[i:i + batch_size]

            try:
                cur.execute(
                    update_query,
                    (validator, batch_words)
                )

                updated = cur.rowcount
                successful += updated
                total_updated += updated

                if updated < len(batch_words):
                    failed += (len(batch_words) - updated)

                print(
                    f"      ✅ Batch {batch_num}/{total_batches} updated "
                    f"({updated}/{len(batch_words)}) for {validator}"
                )

            except Exception as e:
                failed += len(batch_words)
                print(
                    f"      ❌ Batch {batch_num}/{total_batches} failed for {validator}: "
                    f"{str(e)[:100]}"
                )

        stats[validator] = {
            'assigned': successful,
            'failed': failed,
            'total': len(records),
            'valid_assigned': valid_count,
            'invalid_assigned': invalid_count
        }

        print(f"\n      📊 {validator} summary:")
        print(f"         Total assigned: {successful}")
        print(f"         VALID assigned: {valid_count}")
        print(f"         INVALID assigned: {invalid_count}")
        print(f"         Failed: {failed}")

    print(f"\n   💾 Committing all batch updates...")
    conn.commit()
    print(f"   ✅ Commit successful")

    return stats, total_updated


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def generate_assignment_report(assignments, stats):
    """
    Generate detailed assignment report with VALID/INVALID breakdown

    Args:
        assignments: Dict mapping validators to records
        stats: Assignment statistics
    """
    print("\n" + "=" * 70)
    print("📊 VALIDATOR ASSIGNMENT REPORT")
    print("=" * 70)

    total_assigned = sum(s['assigned'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())
    total_valid = sum(s['valid_assigned'] for s in stats.values())
    total_invalid = sum(s['invalid_assigned'] for s in stats.values())

    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   {'─' * 66}")
    print(f"   Total records assigned: {total_assigned}")
    print(f"   VALID records assigned: {total_valid}")
    print(f"   INVALID records assigned: {total_invalid}")
    print(f"   Total failures: {total_failed}")
    success_rate = (total_assigned / (total_assigned + total_failed) * 100) if (
                                                                                           total_assigned + total_failed) > 0 else 0
    print(f"   Success rate: {success_rate:.1f}%")
    print(f"   {'─' * 66}")

    print(f"\n👥 VALIDATOR BREAKDOWN:")
    print(f"   {'─' * 66}")

    for validator in sorted(stats.keys()):
        validator_stats = stats[validator]
        print(f"\n   📌 {validator}:")
        print(f"      Total assigned: {validator_stats['assigned']}")
        print(f"      VALID assigned: {validator_stats['valid_assigned']}")
        print(f"      INVALID assigned: {validator_stats['invalid_assigned']}")
        print(f"      Failed: {validator_stats['failed']}")
        print(f"      Total records: {validator_stats['total']}")

    print(f"\n   {'─' * 66}")
    print("=" * 70)


# ============================================================================
# LOGGING
# ============================================================================

def insert_log(cur, total_assigned, total_failed, validator_stats):
    """Insert log entry into TTC_MASTER_DATA_LOGS"""
    print(f"\n📝 Inserting log into PROCESSED_MASTER_DATA_LOGS...")

    cur.execute("""
        SELECT COALESCE(MAX(CAST(SUBSTRING("LOG_ID" FROM 'LOG_(\\d+)') AS INTEGER)), 0) + 1
        FROM "LOGS"."PROCESSED_MASTER_DATA_LOGS"
    """)
    next_id = cur.fetchone()[0]
    log_id = f"LOG_{next_id:07d}"

    # Create detailed summary for log
    breakdown_parts = []
    for v, s in validator_stats.items():
        breakdown_parts.append(
            f"{v}: {s['assigned']} total (VALID: {s['valid_assigned']}, INVALID: {s['invalid_assigned']})"
        )
    breakdown = ", ".join(breakdown_parts)

    summary = (
        f"Assigned {total_assigned} records to validators. "
        f"Failed: {total_failed}. "
        f"Breakdown: {breakdown}"
    )

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
            "VALIDATOR_ASSIGNMENT_WITH_VALIDITY_SPLIT",
            "MASTER_DATA_LLM_CLASSIFICATION_VALIDATION",
            total_assigned,
            total_failed,
            total_assigned + total_failed,
            datetime.now()
        )
    )

    print(f"   ✅ Log inserted: {log_id}")
    return log_id


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def assign_validators():
    """
    Main function: Assign validators to unassigned LLM classification records
    Distributes VALID and INVALID records separately to ensure balanced assignment
    """

    print("\n" + "=" * 70)
    print("🚀 VALIDATOR ASSIGNMENT - STARTING")
    print("   (Balanced VALID/INVALID Distribution)")
    print("=" * 70)

    conn, cur = connect_to_db()

    try:
        start_time = datetime.now()

        # Get current assignment statistics
        current_stats = get_assignment_statistics(cur)

        # Get available validators
        available_validators = get_available_validators()

        if not available_validators:
            print("\n❌ ERROR: No validators available!")
            print("   Please set at least one validator to is_available=True in VALIDATORS config")
            return

        # Fetch unassigned records by validity
        unassigned_by_validity = fetch_unassigned_records_by_validity(cur)

        valid_records = unassigned_by_validity['VALID']
        invalid_records = unassigned_by_validity['INVALID']

        if not valid_records and not invalid_records:
            print("\n✅ No unassigned records found")
            print("   All records have been assigned to validators!")
            return

        # Distribute VALID records among validators
        valid_assignments = {}
        if valid_records:
            valid_assignments = distribute_records(valid_records, available_validators, 'VALID')

        # Distribute INVALID records among validators
        invalid_assignments = {}
        if invalid_records:
            invalid_assignments = distribute_records(invalid_records, available_validators, 'INVALID')

        # Merge assignments so each validator gets both VALID and INVALID
        combined_assignments = merge_assignments(valid_assignments, invalid_assignments)

        if not combined_assignments:
            print("\n❌ ERROR: Failed to distribute records")
            return

        # Update database with assignments
        validator_stats, total_updated = assign_validators_to_records(cur, conn, combined_assignments)

        # Insert log
        total_assigned = sum(s['assigned'] for s in validator_stats.values())
        total_failed = sum(s['failed'] for s in validator_stats.values())

        log_id = insert_log(cur, total_assigned, total_failed, validator_stats)
        conn.commit()

        # Generate final report
        generate_assignment_report(combined_assignments, validator_stats)

        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n⏱️  EXECUTION TIME: {elapsed:.2f} seconds")
        print(f"📝 Log ID: {log_id}")

        print("\n" + "=" * 70)
        print("✅ VALIDATOR ASSIGNMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)

        # Return stats for potential use in pipeline
        return {
            'total_assigned': total_assigned,
            'total_failed': total_failed,
            'validator_stats': validator_stats,
            'log_id': log_id
        }

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during validator assignment:")
        print(f"   Error: {str(e)}")
        import traceback
        print(f"\n📋 Full traceback:")
        traceback.print_exc()
        conn.rollback()
        print(f"   🔄 Database changes rolled back")
        return None

    finally:
        cur.close()
        conn.close()
        print("\n✅ Database connection closed")


# ============================================================================
# UTILITY FUNCTION FOR VALIDATOR AVAILABILITY MANAGEMENT
# ============================================================================

def set_validator_availability(validator_name, is_available):
    """
    Utility function to enable/disable validators

    Args:
        validator_name: Name of the validator
        is_available: True to enable, False to disable

    Usage:
        set_validator_availability("Madhu Mitha", False)  # Disable Madhu Mitha
        set_validator_availability("Shemanithi", True)    # Enable Shemanithi
    """
    for val_key, val_data in VALIDATORS.items():
        if val_data['name'] == validator_name:
            val_data['is_available'] = is_available
            print(f"✅ {validator_name} availability set to: {is_available}")
            return True

    print(f"❌ Validator '{validator_name}' not found")
    return False


