# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a SAS Econometrics examples repository containing example code and documentation for SAS/ETS procedures, with a focus on:
- HMM (Hidden Markov Model) Procedure
- MARKETATTRIBUTION Procedure (documentation only)

## Key Information

### Technology Stack
- **Language**: SAS (Statistical Analysis System)
- **Product**: SAS Econometrics / SAS/ETS
- **File Types**: `.sas` (SAS code), `.sas7bdat` (SAS data files), `.pdf` (documentation)

### Repository Structure
- `/data/` - SAS data files (.sas7bdat format)
  - `/hmm/` - Data for HMM examples (e.g., finitelearn.sas7bdat)
- `/docs/` - PDF documentation manuals
  - `/hmm/` - HMM procedure manual
  - `/marketattribution/` - Market Attribution procedure manual
- `/examples/` - SAS example code files
  - `/hmm/` - HMM examples (hmmgs.sas, hmmex01.sas through hmmex04.sas)

### Working with SAS Code

1. **No build/test commands**: This is a collection of SAS examples meant to be run in SAS environments. There are no npm, make, or other build commands.

2. **Example code structure**: Each .sas file contains:
   - Header comments with title, description, and purpose
   - Self-contained examples that can be run independently
   - Example 20.4 (hmmex04.sas) uses pretrained data from `/data/hmm/finitelearn.sas7bdat`

3. **Key SAS procedures used**:
   - `PROC HMM` - Main procedure for Hidden Markov Models
   - Standard SAS procedures like DATA steps, ODS graphics

### Important Notes

- All examples are designed to be educational and demonstrate specific econometric techniques
- The repository uses Apache 2.0 License
- SAS code files include comprehensive comments explaining the statistical models being demonstrated
- When modifying examples, maintain the header comment structure and ensure code remains self-contained