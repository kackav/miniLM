import json
import ast
import json_repair


def compute_dst_training_metrics(refs, hyps):
    """
    Compute the metrics for the DST training.
    """
    # Assuming batch_s contains the necessary fields
    assert len(refs) == len(hyps), "References and hypotheses must have the same length"
    ref_dicts = [json.loads(ref) for ref in refs]
    hyp_dicts = []
    for i, hyp in enumerate(hyps):
        try:
            hyp = json_repair.loads(hyp)
        except:
            print(f"Error decoding JSON for hypothesis {i}")
            print(hyp)
        if not(isinstance(hyp, dict)):
            hyp = {}
        hyp_dicts.append(hyp)
    
    # Compute the metrics
    domain_tp = 0
    domain_fp = 0
    domain_fn = 0
    slot_k_tp = 0
    slot_k_fp = 0
    slot_k_fn = 0
    slot_v_tp = 0
    slot_v_fp = 0
    slot_v_fn = 0
    num_erroneous_turns = 0
    ref_labels = []
    hyp_labels = []
    for ref, hyp in zip(ref_dicts, hyp_dicts):
        # Here you can compute your metrics, e.g., accuracy, precision, recall, F1 score
        # For demonstration, we will just print the reference and hypothesis
        this_turn_is_erroneous = False
        ref_trans = ref["label"]
        hyp_trans = hyp.get("label", "")
        ref_labels.append(ref_trans)
        hyp_labels.append(hyp_trans)
        ref_domains = ref["domains"]
        ref_domains = ast.literal_eval(ref_domains)
        hyp_domains = hyp.get("domains", [])
        
        try:
            hyp_domains = ast.literal_eval(hyp_domains)
            if hyp_domains is None:
                hyp_domains = []
        except:
            hyp_domains = []
        for domain in ref_domains:
            if domain in hyp_domains:
                domain_tp += 1
            else:
                domain_fn += 1
                if not this_turn_is_erroneous:
                    num_erroneous_turns += 1
                this_turn_is_erroneous = True

        for domain in hyp_domains:
            if domain not in ref_domains:
                domain_fp += 1
                if not this_turn_is_erroneous:
                    num_erroneous_turns += 1
                this_turn_is_erroneous = True

        ref_slots = ref["slots"]
        ref_slots = ast.literal_eval(ref_slots)
        ref_slots = {k+"_"+v: ref_slots[k][v] for k in ref_slots for v in ref_slots[k]}
        hyp_slots = hyp.get("slots", {})
        try:
            hyp_slots = ast.literal_eval(hyp_slots)
            hyp_slots = {k+"_"+v: hyp_slots[k][v] for k in hyp_slots for v in hyp_slots[k]}
        except:
            hyp_slots = {}
        for slot in ref_slots:
            if slot in hyp_slots:
                slot_k_tp += 1
                if ref_slots[slot] == hyp_slots[slot]:
                    slot_v_tp += 1
                else:
                    slot_v_fn += 1
                    if not this_turn_is_erroneous:
                        num_erroneous_turns += 1
                    this_turn_is_erroneous = True
            else:
                slot_k_fn += 1
                if not this_turn_is_erroneous:
                    num_erroneous_turns += 1
                this_turn_is_erroneous = True
        for slot in hyp_slots:
            if slot not in ref_slots:
                slot_k_fp += 1
                if not this_turn_is_erroneous:
                    num_erroneous_turns += 1
                this_turn_is_erroneous = True
                slot_v_fp += 1 if hyp_slots[slot] else 0
                if not this_turn_is_erroneous and hyp_slots[slot]:
                    num_erroneous_turns += 1
                this_turn_is_erroneous = True

    # Return the computed metrics
    return {
        "domain_tp": domain_tp,
        "domain_fp": domain_fp,
        "domain_fn": domain_fn,
        "slot_k_tp": slot_k_tp,
        "slot_k_fp": slot_k_fp,
        "slot_k_fn": slot_k_fn,
        "slot_v_tp": slot_v_tp,
        "slot_v_fp": slot_v_fp,
        "slot_v_fn": slot_v_fn,
        "num_erroneous_turns": num_erroneous_turns,
        "num_turns": len(refs),
        "ref_labels": ref_labels,
        "hyp_labels": hyp_labels,
    }


def compute_dst_precision_recall_f1(metrics):
    """
    Compute precision, recall, and F1 score from the metrics.
    """
    domain_precision = metrics["domain_tp"] / (metrics["domain_tp"] + metrics["domain_fp"]) if (metrics["domain_tp"] + metrics["domain_fp"]) > 0 else 0
    domain_recall = metrics["domain_tp"] / (metrics["domain_tp"] + metrics["domain_fn"]) if (metrics["domain_tp"] + metrics["domain_fn"]) > 0 else 0
    domain_f1 = 2 * (domain_precision * domain_recall) / (domain_precision + domain_recall) if (domain_precision + domain_recall) > 0 else 0

    slot_k_precision = metrics["slot_k_tp"] / (metrics["slot_k_tp"] + metrics["slot_k_fp"]) if (metrics["slot_k_tp"] + metrics["slot_k_fp"]) > 0 else 0
    slot_k_recall = metrics["slot_k_tp"] / (metrics["slot_k_tp"] + metrics["slot_k_fn"]) if (metrics["slot_k_tp"] + metrics["slot_k_fn"]) > 0 else 0
    slot_k_f1 = 2 * (slot_k_precision * slot_k_recall) / (slot_k_precision + slot_k_recall) if (slot_k_precision + slot_k_recall) > 0 else 0

    slot_v_precision = metrics["slot_v_tp"] / (metrics["slot_v_tp"] + metrics["slot_v_fp"]) if (metrics["slot_v_tp"] + metrics["slot_v_fp"]) > 0 else 0
    slot_v_recall = metrics["slot_v_tp"] / (metrics["slot_v_tp"] + metrics["slot_v_fn"]) if (metrics["slot_v_tp"] + metrics["slot_v_fn"]) > 0 else 0
    slot_v_f1 = 2 * (slot_v_precision * slot_v_recall) / (slot_v_precision + slot_v_recall) if (slot_v_precision + slot_v_recall) > 0 else 0

    return {
        "domain_precision": domain_precision,
        "domain_recall": domain_recall,
        "domain_f1": domain_f1,
        "slot_k_precision": slot_k_precision,
        "slot_k_recall": slot_k_recall,
        "slot_k_f1": slot_k_f1,
        "slot_v_precision": slot_v_precision,
        "slot_v_recall": slot_v_recall,
        "slot_v_f1": slot_v_f1,
    }