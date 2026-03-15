# ELEC0145 — Robotics in Medicine and Industry
## Assignment 2 Brief — Tasks 1 & 2
**Year:** 2025/2026 | **Lecturers:** Dr. Roselina Arelhi & Dr. Yu Wu

---

## Context

You are members of the engineering design team at **Company XYZ**, an automation integrator. A sports rim manufacturer has commissioned your team to propose an automated solution for their post-casting operations. The factory produces **10 different rim types**, with variations in pattern/design, diameter, and width. After casting, rims are placed into a storage area in a largely unstructured/disorganised manner, rather than being neatly indexed or presented in a consistent pose.

The client wants to automate the workflow from this point onward:
- Retrieve a rim from mixed, disorganised storage
- Identify what rim type it is, so the correct programme can be selected
- Deburr and polish the rim
- Transfer the finished rim into an organised storage facility

The solution must combine appropriate sensing, machine vision/learning, handling, robotics, and system integration, while considering practical engineering constraints such as **cycle time/throughput, reliability, safety, maintainability, and cost**.

---

## Task 1: Image Classification
**Page limit: 8 pages** (including text and images)

### Objective
Develop an image-based classifier that identifies the rim type. In the factory scenario, this classification result is used to automatically select the correct deburring and polishing programme in Task 2.

---

### 1.1 Dataset Creation

- Create a small custom dataset containing **10 different rim types (10 classes)**
- **15 different images per class = 150 images total**
- You may define each class in a practical way, for example by using a **specific car model + production year** (e.g., "Porsche 911 2024")

**Image requirements:**
- Aim to reflect the real industrial challenge: rims may appear at **different orientations, distances, and lighting conditions**, and may be **partially occluded**
- For each class, include images that capture this realistic variation
- To avoid the model "cheating" by learning the car body or background rather than the rim design, aim for **rim-focused images** where possible (e.g., crop so the wheel/rim dominates the image)
- **State clearly what preprocessing you applied** (cropping strategy, resizing, etc.)

---

### 1.2 Model Development

Train a **Convolutional Neural Network (CNN)** to classify the 10 rim types. You may either:
- Design a CNN from scratch, or
- Use transfer learning

> You do not need to submit your code, but your report must be detailed enough that another engineer could reproduce your approach.

**The report must include:**
- Software/framework used
- Architecture choice
- Key hyperparameters
- Preprocessing applied
- Training procedure

---

### 1.3 Training, Validation, and Testing

Because the dataset is small, you must explain how you evaluate performance fairly.

**As a minimum, the report must include:**

1. **Data split strategy** — describe and justify (e.g., train/validation/test split or cross-validation)

2. **Final performance metrics** — at minimum:
   - Overall accuracy
   - Confusion matrix — briefly discuss which classes are confused and why

3. **Evidence of training behaviour** — e.g., training/validation curves OR a clear discussion of overfitting and how you mitigated it

4. **At least one robustness technique** — explore and justify (e.g., learning-rate schedule)

5. **If multiple models/settings are compared** — summarise results clearly and explain why you chose your final approach

---

### 1.4 Task 1 Marking Criteria

| Criterion | Weight | Excellent | Good | Satisfactory | Limited |
|-----------|--------|-----------|------|-------------|---------|
| Dataset & preprocessing | 10 | Meets dataset specification. Strong variation. Preprocessing is clear and reproducible. | Meets specification; reasonable variety; preprocessing mostly clear with minor gaps. | Meets specification but limited variety and/or weak justification; preprocessing not fully reproducible. | Partly meets specification or class definitions/strategy unclear. |
| Model development & training approach (incl. robustness) | 15 | Model choice is well-justified for a small dataset. Architecture and training setup are clearly described. Implements and justifies at least one robustness technique. | Sensible approach; most training details included; includes at least one robustness technique but with limited justification or implementation detail. | Basic model/training described but lacks important details and/or robustness technique is mentioned only briefly. | Vague or incomplete model/training description; robustness not meaningfully addressed. |
| Evaluation, overfitting analysis & model selection + industrial reflection | 15 | Evaluation strategy is explained and justified. If multiple models/settings are tried, results are summarised clearly and the final choice is justified. Includes a short industrial reflection (e.g., deployment assumptions and at least one likely failure mode). | Includes required metrics and some interpretation; includes some discussion of overfitting/mitigation or comparisons, but not fully evidenced/clearly linked to results; reflection present but not fully. | Metrics reported but evaluation/analysis is shallow; little or no evidence-based overfitting discussion; reflection generic. | Minimal evaluation evidence; little interpretation; overfitting/mitigation and uncertainty handling unclear. |

---

## Task 2: Automation Process Design
**Page limit: 12 pages** (including text and images)

### Objective
Propose a practical automation concept that covers the **full workflow** from disorganised storage to organised final storage.

---

### 2.1 Required Workflow Steps

**Step 1: Retrieval from disorganised storage**
- Design a method to randomly pick a rim from the disorganised storage area and take it to the deburring station

**Step 2: Identification and programme selection at the deburring station**
- Once the rim reaches the deburring station, the system must select the correct deburring programme based on the rim type
- Explain how the identification result (from Task 1) is used in the control logic

**Step 3: Polishing and programme selection**
- After deburring, the rim must be polished with the correct programme selected based on rim type
- Discuss station layout choice: will one robot perform both deburring and polishing (e.g., tool changer), or will you use separate stations/robots?
- If separated, show how the rim is transferred between stations (conveyor, fixture transfer, etc.)

**Step 4: Access to both sides of the rim (IMPORTANT)**
- Deburring and polishing must occur on **both sides** of the rim
- If your process accesses only one face at a time, you must include a **mechanism or strategy to flip or reorient the rim** so the second side becomes reachable (e.g., flipping station)

**Step 5: Organised storage**
- After both sides are completed, propose a "proper" storage solution (e.g., racks, bins, FIFO lanes)
- Explain how rims are placed, tracked, and retrieved later

---

### 2.2 Design Requirements

In your automation design, you must:
- **Consider several different technologies** for each task and select a technology based on agreed criteria
- **Consider process time and reduce bottlenecks** — shorter hypothetical cycle time will attract higher scores
- **Include sensors** required for the automation
- **Provide an estimated cost** of your final solution

---

### 2.3 Required Deliverables for Task 2

The report must include:
- High-level process diagram (block diagram or flowchart)
- Simple layout sketch of the cell/line
- Brief description of each subsystem
- Timing/cycle-time table
- Cost estimate
- Sensor list
- Safety considerations

---

### 2.4 Task 2 Marking Criteria

| Criterion | Weight | Excellent | Good | Satisfactory | Limited |
|-----------|--------|-----------|------|-------------|---------|
| End-to-end automation concept & completeness | 20 | Coherent workflow supplemented by clear process flow diagram and layout sketch. Both-side processing strategy and final storage explicitly included. | Mostly complete and clear; one element under-specified (e.g. storage, or both-sides handling). | Concept present but hard to follow; diagrams/layout limited; some required elements only mentioned. | Fragmented concept; missing key parts of the workflow. |
| Control logic using Task 1 classifier for programme selection | 15 | Clearly shows how classifier output triggers deburring and polishing programme selection. Includes decision/state logic and low-confidence handling. | Correct mapping and flow; uncertainty handling included but not fully specified. | Mentions classifier-driven selection but logic is shallow; uncertainty handling vague. | Unclear how classification result is used in control. |
| Engineering realism & justification (feasibility) + report organisation (incl. Teamwork section) | 25 | Credible technology choices compared using stated criteria. Includes a cycle-time/timing table, identifies bottlenecks and mitigations. Provides a reasonable cost estimate. Addresses safety, reliability, and maintainability. Report is clearly organised and includes a Teamwork section that explains how tasks were split and integrated. | Covers most engineering items well; minor gaps (e.g., cost high-level, cycle time not tied to bottlenecks, or safety brief). Headings mostly followed; Teamwork section present but brief. | Some engineering justification, but incomplete/weakly supported; limited quantitative support. Structure is inconsistent and/or Teamwork section is superficial. | Mostly qualitative claims; little evidence of feasibility thinking. Report organisation could be improved and/or Teamwork section missing/unclear. |

---

## Report Structure (both tasks)

The report must be organised under these section headings:

1. **Cover page** — Report title, team number, full name + student number + email for each member, submission date
2. **Executive summary**
3. **Task 1: Image Classification**
4. **Task 2: Automation Process**
5. **Conclusion** — Limitations, Recommendations, Future Work
6. **Teamwork** — How tasks were split among team members
7. **References** — IEEE Citation Style, on separate pages

---

## Formatting Requirements

| Rule | Requirement |
|------|------------|
| Font size | **Exactly 11 points** |
| Font type | Calibri or Arial (recommended) |
| Line spacing | Single or Multiple = 1.1 |
| Alignment | Justified (both left and right) |
| Page limit Task 1 | **Max 8 pages** — content beyond limit will NOT be read |
| Page limit Task 2 | **Max 12 pages** — content beyond limit will NOT be read |
| Figures | All must have captions, axis labels, legends where appropriate |
| Curves | Must be distinguishable even if printed in black and white |
| References | IEEE Citation Style throughout |

---

## Additional Notes

- This assignment is **open-ended** — there is no single right or wrong answer
- Your solution will be assessed on how **creative, feasible, and thorough** it is
- Use knowledge from the **IEP Robotics Minor** where appropriate: robot kinematics, image processing, computer vision, object classification, object detection, path planning, sensors and perception
- All third-party sources of information must be **properly credited and referenced in IEEE style**
- Collaboration with other teams via exchange of ideas, sharing of codes, or re-using portions of reports is **not allowed** and will be considered collusion

---
