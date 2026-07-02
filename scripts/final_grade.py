import subprocess
import json
import sys

def run_evaluation():
    print("==================================================")
    print("  SHL AUTOMATED EVALUATION HARNESS (SIMULATED)    ")
    print("==================================================")
    print("\nRunning Evaluation... This may take a moment due to rate limiting fallbacks.\n")

    # 1. Run evaluate_public.py to get Trace Recall and Hard Evals
    print("[1/2] Running Trace Replay (Recall@10 & Hard Evals)...")
    try:
        eval_proc = subprocess.run(
            [sys.executable, "scripts/evaluate_public.py"],
            capture_output=True,
            text=True,
            check=False
        )
        # Find JSON in output (since there might be warning logs)
        lines = eval_proc.stdout.split('\n')
        json_str = ""
        in_json = False
        for line in lines:
            if line.startswith('{'):
                in_json = True
            if in_json:
                json_str += line + "\n"
            if line.startswith('}'):
                break
        
        trace_results = json.loads(json_str)
    except Exception as e:
        print(f"Error parsing trace results: {e}")
        return

    # 2. Run Behavior Probes
    print("[2/2] Running Behavior Probes Suite...")
    try:
        probe_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_behavior.py", "-q", "--tb=no"],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Parse pytest output
        output = probe_proc.stdout
        passed = output.count('.') + output.count('P') # rough count
        failed = output.count('F')
        total = passed + failed
        # Exact parse from summary line if possible
        import re
        match = re.search(r'(\d+) passed', output)
        if match:
            passed = int(match.group(1))
            total = passed
            fail_match = re.search(r'(\d+) failed', output)
            if fail_match:
                failed = int(fail_match.group(1))
                total += failed
    except Exception as e:
        print(f"Error running probes: {e}")
        return

    print("\n\n==================================================")
    print("                FINAL GRADING REPORT              ")
    print("==================================================")

    print("\n1. HARD EVALS (MUST PASS)")
    print("-" * 40)
    print(f"Schema Compliance:      {trace_results.get('schema_pass_rate', 0.0) * 100:>5.1f}%")
    print(f"Catalog Membership:     {trace_results.get('catalog_membership_pass_rate', 0.0) * 100:>5.1f}%")
    print(f"Turn Cap (<8) Honored:  {trace_results.get('turn_cap_pass_rate', 0.0) * 100:>5.1f}%")
    print(f"Zero URLs in Reply:     {'PASS' if trace_results.get('url_in_reply_count', 1) == 0 else 'FAIL'}")
    
    print("\n2. RECALL@10 ON FINAL RECOMMENDATIONS")
    print("-" * 40)
    print(f"Mean Recall@10 (1st Commit): {trace_results.get('mean_first_commit_recall_at_10', 0.0):>5.2f}")
    print(f"Mean Recall@10 (Final):      {trace_results.get('mean_compacted_final_state_recall_at_10', 0.0):>5.2f}")

    print("\n3. BEHAVIOR PROBES")
    print("-" * 40)
    probe_rate = (passed / total) * 100 if total > 0 else 0
    print(f"Total Probes Executed:  {total}")
    print(f"Probes Passed:          {passed}")
    print(f"Probes Failed:          {failed}")
    print(f"Behavior Pass Rate:     {probe_rate:>5.1f}%")
    
    print("\n==================================================")
    
    hard_pass = (
        trace_results.get('schema_pass_rate', 0) == 1.0 and
        trace_results.get('catalog_membership_pass_rate', 0) == 1.0 and 
        trace_results.get('turn_cap_pass_rate', 0) == 1.0 and
        trace_results.get('url_in_reply_count', 1) == 0
    )
    
    print("OVERALL SUBMISSION STATUS: ", end="")
    if hard_pass and probe_rate > 90 and trace_results.get('mean_first_commit_recall_at_10', 0) >= 0.8:
        print("✅ ELIGIBLE FOR TOP TIER")
    elif hard_pass:
        print("✅ PASSING")
    else:
        print("❌ FAILING (Hard evals unmet)")

if __name__ == "__main__":
    run_evaluation()
