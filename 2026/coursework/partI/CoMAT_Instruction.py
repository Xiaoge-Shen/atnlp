'''
Question 3
Create a variable `INSTRUCTION` and fill in the prompt with instructions provided in the file `prompt-instruction.txt.`
'''
from pathlib import Path

# Load the exact prompt from the provided file.
INSTRUCTION = Path(__file__).with_name("prompt-instruction.txt").read_text(encoding="utf-8")
