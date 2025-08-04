## GenAI Proof of Concept: SAS to Python conversion using Claude Code

The purpose of this proof of concept is to find out if an LLM can take an existing SAS codebase and convert it to Python. The project we will be using for this PoC is the sas-econometrics-example project (a SAS econometric libary) from SAS Institute: https://github.com/sassoftware/sas-econometrics-examples

### LLM & AI Tool
* LLM used: Claude Opus 4 (best coding LLM) - https://www.anthropic.com/claude/opus
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

#### Conversion Process: 
* Step 1 - use Claude Code (together with Opus 4 LLM) to analyze an existing code repository, then ask it to put together a comprehensive conversion plan for converting the entire codebase from SAS to Python. 
* Step 2 - ask Claude Code to use this conversion plan (see [PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md)) to implement all phases defined in the plan. Make sure the migration plan includes requirements for comprehensive test coverage of the converted code, via unit and integration tests.

### PoC Results
* The [PYTHON_CONVERSION_PLAN.md](PYTHON_CONVERSION_PLAN.md) specified a migration timeline of 5 weeks. The conversion took Claude Code less than an hour to complete. 
* The conversion effort by Claude Code did not perform a line-by-line conversion of the original SAS code into Python. It analyzed the entire SAS codebase before coming up with a new modular design, then scaffolded the entire project, before proceeding with the SAS to Python conversion.
* The converted Python code resides under python_hmm_examples/ directory
* Successful passing of all unit and integration tests. See [python_hmm_examples/TEST_SUMMARY.md](python_hmm_examples/TEST_SUMMARY.md) for details.

## Running the code
See [python_hmm_examples/README.md](python_hmm_examples/README.md)

## All prompts issued to Claude Code
The complete list of prompts issued to Clause Code is listed below:

> you're a SAS and Python programming language expert. Analyze the existing SAS codebase before coming up with a plan to convert it to Python. Save this plan under PYTHON_CONVERSION_PLAN.md. Think hard.

> Go ahead and implement all tasks in @PYTHON_CONVERSION_PLAN.md. Make sure the converted Python code has comprehensive test coverage, via unit and integration tests.

> Run all tests of the converted Python code and save the test results to TEST_SUMMARY.md
