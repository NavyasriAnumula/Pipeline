from TTC_MERGING_TOOLS import merge_and_classify_pipeline
from TTC_MERGING_TECHNOLOGIES import merge_and_classify_tech_pipeline
from TTC_MERGING_COMPLIANCES import merge_and_classify_compliances_pipeline
from LLM_CLASSIFICATION_VALIDATION import validate_llm_classifications
from VALIDATOR_ASSIGNMENT import assign_validators
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys


# PIPELINE CONFIGURATION
# ============================================================================

PIPELINE_CONFIG = {
    "run_tools_merge": True,
    "run_technologies_merge": True,
    "run_compliances_merge": True,
    "run_validation": True,
    "run_validator_assignment": True,
    "stop_on_error": False
}


# ============================================================================
# STEP RUNNER HELPER
# ============================================================================

def run_step(name, fn):
    """Runs a pipeline step function and returns a result dict."""
    print(f"\n▶️  [{name}] Starting...")
    step_start = datetime.now()
    try:
        fn()
        elapsed = (datetime.now() - step_start).total_seconds()
        print(f"\n✅ [{name}] Completed in {elapsed:.2f}s")
        return name, {"status": "SUCCESS", "elapsed_seconds": elapsed}
    except Exception as e:
        elapsed = (datetime.now() - step_start).total_seconds()
        print(f"\n❌ [{name}] Failed after {elapsed:.2f}s — {str(e)}")
        return name, {"status": "FAILED", "elapsed_seconds": elapsed, "error": str(e)}


# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_pipeline():
    """Execute the complete TTC data processing pipeline"""

    print("\n" + "=" * 80)
    print("🚀 TTC DATA PROCESSING PIPELINE - STARTING")
    print("=" * 80)
    print(f"\nPipeline Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPipeline Steps:")
    print("  1+2+3. Merge & Classify TOOLS + TECHNOLOGIES + COMPLIANCES (parallel)")
    print("  4.     Validate LLM Classifications (DeepSeek)")
    print("  5.     Assign Validators")
    print("\n" + "=" * 80)

    pipeline_start = datetime.now()
    results = {}
    errors = []

    # ========================================================================
    # STEPS 1, 2, 3: RUN IN PARALLEL
    # ========================================================================
    print("\n" + "█" * 80)
    print("STEPS 1+2+3: TOOLS / TECHNOLOGIES / COMPLIANCES — RUNNING IN PARALLEL")
    print("█" * 80)

    parallel_steps = []
    if PIPELINE_CONFIG["run_tools_merge"]:
        parallel_steps.append(("TOOLS MERGE", merge_and_classify_pipeline))
    if PIPELINE_CONFIG["run_technologies_merge"]:
        parallel_steps.append(("TECHNOLOGIES MERGE", merge_and_classify_tech_pipeline))
    if PIPELINE_CONFIG["run_compliances_merge"]:
        parallel_steps.append(("COMPLIANCES MERGE", merge_and_classify_compliances_pipeline))

    if parallel_steps:
        with ThreadPoolExecutor(max_workers=len(parallel_steps)) as executor:
            futures = {
                executor.submit(run_step, name, fn): name
                for name, fn in parallel_steps
            }
            for future in as_completed(futures):
                name, result = future.result()
                results[name] = result
                if result["status"] == "FAILED":
                    errors.append(f"{name} failed: {result.get('error', 'Unknown')}")

        # Check stop_on_error after all parallel steps finish
        if errors and PIPELINE_CONFIG["stop_on_error"]:
            print("\n🛑 Pipeline stopped due to error (stop_on_error=True)")
            _print_summary(results, errors, pipeline_start)
            return results, errors
    else:
        print("\n⏭️  All merge steps are SKIPPED (disabled in config)")

    # ========================================================================
    # STEP 4: LLM CLASSIFICATION VALIDATION (sequential)
    # ========================================================================
    if PIPELINE_CONFIG["run_validation"]:
        print("\n" + "█" * 80)
        print("STEP 4: LLM CLASSIFICATION VALIDATION USING DEEPSEEK")
        print("█" * 80)
        print("\n📌 Validating classifications from Steps 1-3")

        _, result = run_step("VALIDATION", validate_llm_classifications)
        results["VALIDATION"] = result
        if result["status"] == "FAILED":
            errors.append(f"VALIDATION failed: {result.get('error', 'Unknown')}")
            if PIPELINE_CONFIG["stop_on_error"]:
                print("\n🛑 Pipeline stopped due to error (stop_on_error=True)")
                _print_summary(results, errors, pipeline_start)
                return results, errors
    else:
        print("\n⏭️  STEP 4 (Validation) - SKIPPED")

    # ========================================================================
    # STEP 5: VALIDATOR ASSIGNMENT (sequential)
    # ========================================================================
    if PIPELINE_CONFIG["run_validator_assignment"]:
        print("\n" + "█" * 80)
        print("STEP 5: VALIDATOR ASSIGNMENT")
        print("█" * 80)
        print("\n📌 Assigning validators to unassigned records")
        print("   (Excluding IS_IRRELEVANT=Yes only records)")

        step_start = datetime.now()
        try:
            assignment_result = assign_validators()
            elapsed = (datetime.now() - step_start).total_seconds()

            if assignment_result:
                results["VALIDATOR ASSIGNMENT"] = {
                    "status": "SUCCESS",
                    "elapsed_seconds": elapsed,
                    "total_assigned": assignment_result.get("total_assigned", 0),
                    "total_failed": assignment_result.get("total_failed", 0),
                    "validator_stats": assignment_result.get("validator_stats", {}),
                    "log_id": assignment_result.get("log_id", ""),
                }
                print(f"\n✅ STEP 5 COMPLETED in {elapsed:.2f}s")
                print(f"   📊 Assigned: {assignment_result.get('total_assigned', 0)} records")
                print(f"   ❌ Failed:   {assignment_result.get('total_failed', 0)} records")
            else:
                results["VALIDATOR ASSIGNMENT"] = {
                    "status": "NO_ACTION",
                    "elapsed_seconds": elapsed,
                    "message": "No records to assign or no validators available",
                }
                print(f"\n⏭️  STEP 5 COMPLETED (No Action) in {elapsed:.2f}s")

        except Exception as e:
            elapsed = (datetime.now() - step_start).total_seconds()
            error_msg = f"STEP 5 (Validator Assignment) failed: {str(e)}"
            errors.append(error_msg)
            results["VALIDATOR ASSIGNMENT"] = {"status": "FAILED", "elapsed_seconds": elapsed, "error": str(e)}
            print(f"\n❌ {error_msg}")
            if PIPELINE_CONFIG["stop_on_error"]:
                print("\n🛑 Pipeline stopped due to error (stop_on_error=True)")
    else:
        print("\n⏭️  STEP 5 (Validator Assignment) - SKIPPED")

    _print_summary(results, errors, pipeline_start)
    return results, errors


# ============================================================================
# SUMMARY PRINTER
# ============================================================================

def _print_summary(results, errors, pipeline_start):
    pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()

    print("\n" + "=" * 80)
    print("📊 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\n⏱️  Total Execution Time: {pipeline_elapsed:.2f}s ({pipeline_elapsed / 60:.1f} minutes)")
    print(f"\n📈 STEP RESULTS:")
    print(f"   {'─' * 76}")

    for step_name, step_result in results.items():
        status = step_result["status"]
        icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⏭️"
        print(f"   {icon} {step_name}: {status}  ({step_result.get('elapsed_seconds', 0):.2f}s)")
        if status == "FAILED":
            print(f"      Error: {step_result.get('error', '')[:80]}")
        if step_name == "VALIDATOR ASSIGNMENT" and "total_assigned" in step_result:
            print(f"      Assigned: {step_result['total_assigned']}  |  Failed: {step_result.get('total_failed', 0)}")
            for validator, stats in step_result.get("validator_stats", {}).items():
                print(f"         {validator}: {stats.get('assigned', 0)}")

    print(f"   {'─' * 76}")
    if errors:
        print(f"\n❌ ERRORS ENCOUNTERED: {len(errors)}")
        for idx, err in enumerate(errors, 1):
            print(f"   {idx}. {err}")
        print("\n⚠️  Pipeline completed WITH ERRORS")
    else:
        print("\n✅ Pipeline completed SUCCESSFULLY")

    print(f"\nPipeline End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("   TTC DATA PROCESSING PIPELINE")
    print("   Tools + Technologies + Compliances (parallel) → Validation → Assignment")
    print("🚀" * 40 + "\n")

    try:
        results, errors = run_pipeline()
        sys.exit(1 if errors else 0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ CRITICAL PIPELINE FAILURE:\n   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
