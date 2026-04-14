# OOB Testing System for NumPy Module

## Project Goal
The goal of this project is to design and implement a simplified OOB (Out Of The Box) testing system for the **NumPy** library – the fundamental package for scientific computing with Python.

The project focuses on configuring a CI/CD process (GitHub Actions), creating functional and performance tests, and developing acceptance test scenarios.

As part of the project, the pipeline is (or will be) designed to:
* Download the latest stable commit from the official NumPy GitHub repository.
* Compile and build the module locally in the runner environment.
* Thoroughly test the built version using our defined pipeline.

---

## Quick Start
**pip install -r requirements.txt**\
Then trigger the pipeline via GitHub Actions (**workflow_dispatch**)\
or locally: **pytest tests/ -v**

---

## Team and Roles
The project is carried out in a 3-person team:

| Name and Surname    | Team Role                                |
|:--------------------|:-----------------------------------------|
| [Adrian Zając]      | **Project Manager / Test / CI Engineer** |
| [Tytus Barański]    | **Test / CI Engineer**                   |
| [Franciszek Latała] | **Test / CI Engineer**                   |

---

## Communication Channels
The main communication channels in the team are:
* **Messenger** – daily communication, quick problem solving, and synchronization meetings.
* **GitHub (Issues & Pull Requests)** – task management, ticket assignment, code review, and history of project agreements.
Weekly teams meetings on Messenger every:
* *Monday* from 17:00 to 18:00 (GMT +2) - prepare work plan for given week
* *Friday* from 7:20 to 9:55 (GMT +2) - summarise work done in given week, share insights and problems

---

## Project Schedule

1. **Week 1 (02.03.2026 - 08.03.2026)**
   - [x] Setting up the repository.
   - [x] Creating a README file with the goal and description.
   - [x] Defining team roles and communication channels.
   - [x] Planning initial test scenarios.
2. **Week 2 (09.03.2026 - 15.03.2026)**
   - [x] Setting up the download of the latest stable commit from the official NumPy Github repo
   - [x] Setting up the local compilation and build of the module
3. **Week 3-4 (16.03.2026 - 29.03.2026)**
   - [x] Implementing performance tests
4. **Week 5-9 (30.03.2026 - 03.05.2026)**
   - Implementing functional tests
5. **Week 10 (04.05.2026 - 10.05.2026)**
   - Implementing acceptance tests
   - Project finalization

---

## 📂 Initial Directory Structure
```text
.
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml     # Configuration of the GitHub Actions pipeline
├── docs/
│   └── test_scenarios.md       # Detailed description of acceptance test scenarios
├── tests/
│   ├── functional/             # Functional test scripts
│   └── performance/            # Performance test scripts testing NumPy
├── scripts/
│   └── build_numpy.py          # Script to locally build NumPy from source
├── requirements.txt            # Dependencies needed to run the tests
└── README.md                   # This file
```

---

## Definition of Done (DoD)
A task or milestone is considered done when all of the following conditions are met:

**Code & Tests**
* The code runs without errors in the CI pipeline on main
* All existing tests continue to pass after the change
* New functionality is covered by at least one test

**Version Control**
* The change was developed on a separate feature branch (e.g. feature/week-X)
* The branch was merged into main via a Pull Request
* The PR was reviewed by at least one other team member before merging
* All commit messages are written in the imperative mood and are under ~50 characters

**Documentation**
* The README accurately reflects the current state of the project
* If the change introduces new test scenarios, docs/test_scenarios.md is updated accordingly

**Pipeline**

* The GitHub Actions pipeline runs successfully on main after the merge
* Benchmark results (if applicable) are saved and visible in the pipeline output

---
