# Proof-of-concept: SAS to Python migration using Claude Code
sas-econometrics-example source repo is located at https://github.com/sassoftware/sas-econometrics-examples

This POC is to evaluate Claude Code (an agentic coding tool from Anthropic: https://www.anthropic.com/claude-code) for its ability to convert a modeling library written in SAS to Python.

#### Conversion Process: 
* Step 1 - use a reasoning LLM that's able to analyze an existing code repository, then put together a comprehensive migration plan for converting the entire project's codebase from SAS to Python. We used Anthropic's Claude Opus 4 LLM for our reasoning LLM. We chose Opus 4 over OpenAI's ChatGPT o3 (advanded reasoning) and Google Gemini 2.5 Pro (reasoning) due to its advanced ability to analyze code. 
* Step 2 - use this migration plan (see migration_plan.md) with Claude Code (together with Claude Opus 4 LLM, known as the most advanded model for agentic coding tasks) to implement all tasks in all phases defined in the migration plan. The migration plan includes requirements for comprehensive code coverage via unit and integration testing.

The conversion took Claude Code about 1 hour to complete. This includes the successful passing of all unit and integration tests. See python_hmm_examples/TEST_SUMMARY.md for details. The converted python codebase resides under python_hmm_examples folder.


## Running the code
See python_hmm_examples/README.md
