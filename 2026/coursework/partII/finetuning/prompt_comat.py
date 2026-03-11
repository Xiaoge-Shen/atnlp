import re

COMAT_INSTRUCTION = (
    "Solve the math problem using a structured approach: "
    "(1) [Identify & Define] List all variables (unknowns) and constants (known values). "
    "(2) [Structural Logic] Write the key mathematical equations or relationships. "
    "(3) [Explicit Facts] Substitute in all the known numeric values. "
    "(4) [Solve] Perform the computation step by step. "
    "Provide the final answer as 'the answer is [answer]' format."
)


def _build_comat_completion(question, reasoning, final_ans):
    """
    Programmatically wraps the existing GSM8K reasoning in CoMAT-style headers.

    s1 - Identify & Define: what variable we are solving for (inferred from question)
    s2 - Structural Logic:   first equation-like line from the reasoning
    s3 - Explicit Facts:     numeric values extracted from the question
    s4 - Solve:              the original GSM8K step-by-step reasoning
    """
    # --- s1: Identify & Define ---
    q_lower = question.lower()
    find_match = re.search(
        r"how (?:many|much|long|far|old|tall|big|large|heavy|wide|deep|fast|soon|often)\s+([\w\s]+?)\??$",
        q_lower,
    )
    if find_match:
        target = find_match.group(1).strip().rstrip("?").strip()
        s1 = f"Let x = the number of {target} we need to find."
    else:
        s1 = "Let x = the unknown quantity to be determined."

    # --- s2: Structural Logic ---
    # Pick the first line in the reasoning that looks like an equation or arithmetic expression
    equation_lines = [
        line.strip()
        for line in reasoning.split("\n")
        if re.search(r"[=+\-*/]", line) and line.strip()
    ]
    if equation_lines:
        s2 = equation_lines[0][:150]
    else:
        s2 = "Set up equations based on the relationships described in the problem."

    # --- s3: Explicit Facts ---
    # Extract "number + context" pairs from the question
    number_pattern = r"(\$?[\d,]+\.?\d*)\s+([\w\s]{1,30}?)(?=[,\.\n?]|$)"
    raw_facts = re.findall(number_pattern, question)
    facts_lines = []
    seen = set()
    for num, ctx in raw_facts[:6]:
        ctx = ctx.strip().rstrip("and").strip()
        entry = f"  - {num} {ctx}".rstrip()
        if entry not in seen and ctx:
            facts_lines.append(entry)
            seen.add(entry)
    s3 = "\n".join(facts_lines) if facts_lines else "  - (see problem statement)"

    completion = (
        f"[Identify & Define] {s1}\n"
        f"[Structural Logic] {s2}\n"
        f"[Explicit Facts]\n{s3}\n"
        f"[Solve]\n{reasoning}\n\n"
        f"The answer is {final_ans}"
    )
    return completion


def comat_formatting_prompts_func(tokenizer, example):
    q = example["question"].strip()
    raw_answer = example["answer"].strip()

    if "####" in raw_answer:
        reasoning, final_ans = raw_answer.split("####", 1)
        reasoning = reasoning.strip()
        final_ans = final_ans.strip()
    else:
        reasoning = raw_answer
        final_ans = ""

    formatted_answer = _build_comat_completion(q, reasoning, final_ans)

    messages = [
        {"role": "system", "content": COMAT_INSTRUCTION},
        {"role": "user", "content": q},
        {"role": "assistant", "content": formatted_answer},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    formatted_text += tokenizer.eos_token

    return {"text": formatted_text}
