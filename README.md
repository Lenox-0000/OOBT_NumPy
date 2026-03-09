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

| Name and Surname | Team Role | Main Responsibilities |
| :--- | :--- | :--- |
| [Adrian Zając] | **Project Manager / Tester** | Work coordination, schedule, code review, acceptance tests. |
| [Tytus Barański] | **DevOps / CI Engineer** | GitHub Actions configuration, building module from source, resolving build issues. |
| [Franciszek Latała] | **QA / Test Engineer** | Implementation of functional tests (pytest), performance tests, results analysis. |

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
   - Setting up the download of the latest stable commit from the official NumPy Github repo
   - Setting up the local compilation and build of the module
3. **Week 3-4 (16.03.2026 - 29.03.2026)**
   -Implementing performance tests
4. **Week 5-9 (30.03.2026 - 03.05.2026)**
   -Implementing functional tests
5. **Week 10 (04.05.2026 - 10.05.2026)**
   -Implementing acceptance tests
   -Project finalization

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

## Initial Test Scenarios (Test Strategy)

All detailed steps and acceptance criteria for acceptance test scenarios will be placed in the `docs/test_scenarios.md` document. 
Outline the test strategy:

### 1. Functional Tests (3-5 scenarios)
Focus on verifying the use of fundamental NumPy functionalities:
* **Matrix Operations (Matrix Multiplication)**: Verifying the correctness of multiplication results (e.g., `np.dot` / `@`) for different data types.
* **Aggregations and Statistics**: Verifying functions such as `np.mean`, `np.median`, `np.std` for handling data with edge values (e.g., NaN).
* **3D Dimension Manipulation**: Testing the `reshape`, `transpose` functions, and correct memory layout after operations.
* **Broadcasting**: Validating broadcasting rules for arrays with different (but compatible) shapes.

### 2. Performance Tests (1-2 scenarios)
Simple execution time measurements (benchmarks):
* **Comparing Native Lists with NumPy**: Time to assign thousands of mathematical operations (element-wise) on regular Python structures compared to optimized C operations in NumPy.
* **Large Matrix Multiplications**: Measuring the time of operations on matrices of sizes e.g., 1000x1000 or larger, saving the result to logs to track regression.

### 3. Acceptance Tests
3 high-level scenarios validating the environment:
1. Successful installation (or compilation) and importing of the module.
2. Running the full set of functional and performance tests reporting 100% satisfactory results.
3. Reporting system - whether the generated results (logs/console) are readable after manually triggering the pipeline.
