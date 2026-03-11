"""
utils_comat.py — Model evaluation utility using the CoMAT system instruction (Q5).

Drop-in replacement for utils.py that uses the CoMAT structured instruction
instead of the generic "Think step by step" prompt.
"""

import torch

COMAT_INSTRUCTION = (
    "Solve the math problem using a structured approach: "
    "(1) [Identify & Define] List all variables (unknowns) and constants (known values). "
    "(2) [Structural Logic] Write the key mathematical equations or relationships. "
    "(3) [Explicit Facts] Substitute in all the known numeric values. "
    "(4) [Solve] Perform the computation step by step. "
    "Provide the final answer as 'the answer is [answer]' format."
)


def model_evaluation_comat(model, tokenizer, system_content, question, max_new_tokens):
    """
    Same interface as the original model_evaluation() in utils.py,
    but uses COMAT_INSTRUCTION as the system prompt.
    """
    messages = [
        {"role": "system", "content": COMAT_INSTRUCTION},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    input_ids = inputs.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=None,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    model_result = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return model_result
