"""Execute a declared program universe and evaluate unseen inputs independently."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


class UndefinedResult(Exception):
    """A legal program has no defined answer on this input."""


def execute(program, example):
    """Interpret pure filter/relate/unique/count/exist/query scene programs."""
    scene = example["scene"]
    objects = scene["objects"]
    slots = example.get("slots", {})

    def value(spec):
        if isinstance(spec, dict) and set(spec) == {"slot"}:
            if spec["slot"] not in slots:
                raise UndefinedResult("Missing question slot")
            return slots[spec["slot"]]
        return spec

    def object_set(result):
        if not isinstance(result, list) or any(type(i) is not int or not 0 <= i < len(objects) for i in result):
            raise UndefinedResult("Expected a set of object identifiers")
        return sorted(set(result))

    def evaluate(node):
        op = node["op"]
        arguments = [evaluate(item) for item in node.get("inputs", [])]
        if op == "scene" and not arguments:
            return list(range(len(objects)))
        if op == "filter" and len(arguments) == 1:
            attribute, wanted = value(node["attribute"]), value(node["value"])
            return [i for i in object_set(arguments[0]) if objects[i].get(attribute) == wanted]
        if op == "unique" and len(arguments) == 1:
            ids = object_set(arguments[0])
            if len(ids) != 1:
                raise UndefinedResult("Unique requires exactly one object")
            return ids[0]
        if op == "relate" and len(arguments) == 1:
            i = arguments[0]
            if type(i) is not int or not 0 <= i < len(objects):
                raise UndefinedResult("Relate requires an object")
            relation = value(node["relation"])
            if relation not in scene.get("relationships", {}):
                raise UndefinedResult("Unavailable scene relation")
            return object_set(scene["relationships"][relation][i])
        if op in {"count", "exist"} and len(arguments) == 1:
            ids = object_set(arguments[0])
            return len(ids) if op == "count" else bool(ids)
        if op == "query" and len(arguments) == 1:
            i, attribute = arguments[0], value(node["attribute"])
            if type(i) is not int or not 0 <= i < len(objects) or attribute not in objects[i]:
                raise UndefinedResult("Query requires an object with the requested attribute")
            return objects[i][attribute]
        if op == "equal" and len(arguments) == 2:
            return type(arguments[0]) is type(arguments[1]) and arguments[0] == arguments[1]
        raise ValueError("Unsupported operator or arity: " + str(op))

    try:
        return evaluate(program)
    except UndefinedResult:
        return None


def interpreter(path=None):
    if path is None:
        return execute
    spec = importlib.util.spec_from_file_location("ecs_local_interpreter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.execute


def key(answer):
    return json.dumps(answer, sort_keys=True, separators=(",", ":"), allow_nan=False)


def entropy(masses):
    values = np.asarray(masses, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())


def acquire(problem, label_oracle, run_program=execute, variant="full"):
    """Fit only on the declared reservoir; no held-out inputs are accepted."""
    programs, reservoir = problem["programs"], problem["reservoir"]
    if variant not in {"full", "uncompressed", "fixed_probes"}:
        raise ValueError("Unknown acquisition variant")
    if not isinstance(problem["execution_budget"], int) or problem["execution_budget"] < 0 or not np.isfinite(problem["label_budget"]) or problem["label_budget"] < 0:
        raise ValueError("Execution and label budgets must be nonnegative")
    if not 0 <= problem.get("exposure_ratio", .5) <= 1:
        raise ValueError("exposure_ratio must lie in [0,1]")
    weights = np.asarray(problem.get("prior", np.ones(len(programs))), dtype=float)
    if weights.shape != (len(programs),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Prior must be finite, nonnegative and nonempty")
    weights /= weights.sum()
    costs = np.asarray(problem["query_cost"], dtype=float)
    if costs.shape != (len(reservoir),) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Each reservoir input needs a positive finite label cost")
    order = problem.get("probe_order", list(range(len(reservoir))))
    if sorted(order) != list(range(len(reservoir))):
        raise ValueError("probe_order must be a predeclared reservoir permutation")
    probes, cache, execution_ledger, label_ledger, history = [], {}, [], [], []
    used = 0
    billed = 0.
    queried = set()

    def expand(index):
        nonlocal used
        active = np.flatnonzero(weights > 0)
        if used + len(active) > problem["execution_budget"]:
            return False
        outputs = {}
        for program in active:
            answer = run_program(programs[program], reservoir[index])
            cache[int(program), index] = key(answer)
            outputs[str(program)] = answer
            used += 1
        probes.append(index)
        execution_ledger.append({"probe": index, "program_outputs": outputs, "calls": len(active)})
        return True

    def partition():
        classes = {}
        for program in np.flatnonzero(weights > 0):
            signature = tuple(cache[int(program), probe] for probe in probes)
            classes.setdefault(signature, []).append(int(program))
        return list(classes.values())

    for probe in dict.fromkeys(problem.get("initial_probes", [0])):
        if not 0 <= probe < len(reservoir) or not expand(probe):
            raise ValueError("Initial probe bank is invalid or exceeds the execution budget")
    while billed < problem["label_budget"]:
        classes = partition()
        masses = [float(weights[members].sum()) for members in classes]
        total, exposed = entropy(weights), entropy(masses)
        history.append({"classes": classes, "class_mass": masses, "total_entropy_bits": total,
                        "exposed_entropy_bits": exposed, "hidden_entropy_bits": total - exposed,
                        "executions": used, "label_cost": billed})
        if total < 1e-12:
            break
        while variant != "fixed_probes" and exposed < problem.get("exposure_ratio", .5) * total:
            probe = next((index for index in order if index not in probes), None)
            if probe is None or not expand(probe):
                break
            classes = partition()
            exposed = entropy([float(weights[members].sum()) for members in classes])
        available = [probe for probe in probes if probe not in queried and billed + costs[probe] <= problem["label_budget"]]
        if not available:
            break
        utilities = []
        for probe in available:
            distribution = {}
            members = [[int(i)] for i in np.flatnonzero(weights > 0)] if variant == "uncompressed" else classes
            for group in members:
                answer = cache[group[0], probe]
                distribution[answer] = distribution.get(answer, 0.) + float(weights[group].sum())
            utilities.append(entropy(list(distribution.values())) / costs[probe])
        probe = available[int(np.argmax(utilities))]
        answer = label_oracle(probe)
        encoded = key(answer)
        queried.add(probe)
        billed += float(costs[probe])
        for program in np.flatnonzero(weights > 0):
            if cache[int(program), probe] != encoded:
                weights[program] = 0
        if weights.sum() == 0:
            label_ledger.append({"probe": probe, "answer": answer, "cost": float(costs[probe]), "empty_support": True})
            break
        weights /= weights.sum()
        label_ledger.append({"probe": probe, "answer": answer, "cost": float(costs[probe]), "empty_support": False})
    return {"programs": programs, "posterior": weights.tolist(), "probes": probes,
            "classes": partition() if weights.sum() else [], "history": history,
            "execution_ledger": execution_ledger, "label_ledger": label_ledger,
            "execution_cost": used, "label_cost": billed, "variant": variant}


def evaluate(posterior, examples, run_program=execute, execution_budget=None):
    """Execute every surviving member on unseen inputs and keep all denominators."""
    weights = np.asarray(posterior["posterior"], dtype=float)
    if weights.shape != (len(posterior["programs"]),) or not np.isfinite(weights).all() or (weights < 0).any() or (weights.sum() > 0 and not np.isclose(weights.sum(), 1.)):
        raise ValueError("Posterior must have finite normalized program weights, or empty support")
    if execution_budget is not None and (not isinstance(execution_budget, int) or execution_budget < 0):
        raise ValueError("Held-out execution budget must be a nonnegative integer")
    active = np.flatnonzero(weights > 0)
    records, calls = [], 0
    for index, example in enumerate(examples):
        distribution, outputs = {}, {}
        complete = len(active) > 0 and (execution_budget is None or calls + len(active) <= execution_budget)
        if complete:
            for program in active:
                answer = run_program(posterior["programs"][program], example["input"])
                encoded = key(answer)
                distribution[encoded] = distribution.get(encoded, 0.) + float(weights[program])
                outputs[str(program)] = answer
                calls += 1
        prediction = json.loads(min(distribution, key=lambda answer: (-distribution[answer], answer))) if distribution else None
        correct = complete and prediction is not None and key(prediction) == key(example["answer"])
        records.append({"id": example.get("id", index), "prediction": prediction, "reference": example["answer"],
                        "correct": bool(correct), "complete_execution": complete, "program_outputs": outputs,
                        "distribution": [{"answer": json.loads(answer), "mass": mass} for answer, mass in sorted(distribution.items())]})
    return {"accuracy": sum(row["correct"] for row in records) / len(records) if records else None,
            "examples": len(records), "undefined_or_unexecuted": sum(row["prediction"] is None for row in records),
            "execution_cost": calls, "records": records}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("acquire")
    fit.add_argument("--problem", required=True)
    fit.add_argument("--labels", required=True)
    fit.add_argument("--variant", choices=["full", "uncompressed", "fixed_probes"], default="full")
    test = sub.add_parser("evaluate")
    test.add_argument("--posterior", required=True)
    test.add_argument("--heldout", required=True)
    test.add_argument("--execution-budget", type=int)
    for command in [fit, test]:
        command.add_argument("--interpreter", help="Local Python module exporting execute(program,input); omit for the built-in scene interpreter")
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    run_program = interpreter(args.interpreter)
    if args.command == "acquire":
        problem = json.loads(Path(args.problem).read_text(encoding="utf-8"))
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        result = acquire(problem, lambda index: labels[index], run_program, args.variant)
    else:
        # Held-out inputs and references are opened only in this separate evaluation command.
        posterior = json.loads(Path(args.posterior).read_text(encoding="utf-8"))
        examples = json.loads(Path(args.heldout).read_text(encoding="utf-8"))
        result = evaluate(posterior, examples, run_program, args.execution_budget)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
