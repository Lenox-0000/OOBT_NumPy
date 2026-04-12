# OOB Testing System for NumPy Module

## Project Goal
The goal of this project is to design and implement a simplified OOB (Out Of The Box) testing system for the **NumPy** library – the fundamental package for scientific computing with Python.

The project focuses on configuring a CI/CD process (GitHub Actions), creating functional and performance tests, and developing acceptance test scenarios.

As part of the project, the pipeline is (or will be) designed to:
* Download the latest stable commit from the official NumPy GitHub repository.
* Compile and build the module locally in the runner environment.
* Thoroughly test the built version using our defined pipeline.

---

## Team and Roles (To Be Revised)
The project is carried out in a 3-person team:

| Name and Surname    | Team Role                    |
|:--------------------|:-----------------------------|
| [Adrian Zając]      | **Project Manager / Tester** |
| [Tytus Barański]    | **Test / CI Engineer**       |
| [Franciszek Latała] | **Test / CI Engineer**       |

---

## Communication Channels
The main communication channels in the team are:
* **Messenger** – daily communication, quick problem solving, and synchronization meetings.
* **GitHub (Issues & Pull Requests)** – task management, ticket assignment, code review, and history of project agreements.

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
