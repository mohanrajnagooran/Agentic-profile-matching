"""
generate_sample_data.py — create 100 realistic synthetic resumes and a sample JD.

    python generate_sample_data.py            # writes data/resumes/*.txt and data/job_descriptions/*.txt

Deterministic (seeded) so rankings are reproducible across runs and in tests.
Two hand-written resumes (John Smith, Jane Doe) are included so the assignment's
example query "Why did John rank higher than Jane?" works out of the box.
"""
from __future__ import annotations

import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES_DIR = BASE / "data" / "resumes"
JD_DIR = BASE / "data" / "job_descriptions"
CURRENT_YEAR = 2026

FIRST = ["Aarav", "Priya", "Liam", "Olivia", "Noah", "Emma", "Rohan", "Ananya", "Mateo", "Sofia",
         "Ethan", "Mia", "Arjun", "Kavya", "Lucas", "Chloe", "Vikram", "Isha", "Daniel", "Grace",
         "Karthik", "Meera", "Omar", "Leila", "Hiroshi", "Yuki", "Diego", "Valentina", "Samuel", "Zara",
         "Aditya", "Divya", "Felix", "Hannah", "Ravi", "Sneha", "Marcus", "Nina", "Tariq", "Fatima",
         "Kenji", "Aiko", "Pedro", "Lucia", "Nikhil", "Pooja", "Ben", "Ava", "Sanjay", "Lakshmi"]
LAST = ["Sharma", "Patel", "Nguyen", "Garcia", "Kim", "Iyer", "Brown", "Martinez", "Chen", "Reddy",
        "Wilson", "Kumar", "Lopez", "Singh", "Taylor", "Nair", "Anderson", "Rao", "Thomas", "Menon",
        "Moore", "Das", "Jackson", "Pillai", "White", "Ito", "Harris", "Silva", "Clark", "Khan"]
CITIES = ["Chennai", "Bengaluru", "Hyderabad", "Pune", "Mumbai", "Remote", "San Francisco", "London",
          "Berlin", "Toronto", "Singapore", "Austin"]
COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Labs", "Hooli", "Stark Digital", "Wayne Systems",
             "Cyberdyne", "Soylent Tech", "Pied Piper", "Vandelay Industries", "Massive Dynamic",
             "Tyrell Software", "Oscorp", "Wonka Analytics", "Nakatomi Cloud", "Zoho-like Co",
             "FinEdge", "HealthBridge", "ShopSphere", "LogiTrack", "EduNova", "PayStream", "Medisync"]
DEGREES = ["B.Tech Computer Science, Anna University", "B.E. Information Technology, VIT",
           "B.S. Computer Science, University of Texas", "M.S. Computer Science, Georgia Tech",
           "M.Tech Software Engineering, IIT Madras", "B.Sc Mathematics, Delhi University",
           "MBA, IIM Bangalore", "B.Tech Electronics, NIT Trichy", "Coding Bootcamp, General Assembly",
           "PhD Computer Science, Stanford University", "B.A. Economics, University of Mumbai"]
CERTS = {"AWS": "AWS Certified Solutions Architect – Associate", "Kubernetes": "Certified Kubernetes Administrator (CKA)",
         "Azure": "Microsoft Azure Developer Associate", "GCP": "Google Cloud Professional Developer",
         "Machine Learning": "TensorFlow Developer Certificate", "Agile": "Certified ScrumMaster (CSM)"}

# archetype -> (titles by seniority, core skills, secondary skills)
ARCHETYPES = {
    "fullstack_js": (["Full Stack Developer", "Full Stack Engineer", "Senior Full Stack Engineer", "Lead Full Stack Engineer"],
                     ["React", "TypeScript", "Node.js", "JavaScript", "REST APIs"],
                     ["Express", "GraphQL", "AWS", "Docker", "PostgreSQL", "MongoDB", "Redux", "Jest", "Next.js", "Kubernetes", "CI/CD"]),
    "frontend": (["Frontend Developer", "Frontend Engineer", "Senior Frontend Engineer", "Staff Frontend Engineer"],
                 ["React", "JavaScript", "HTML/CSS", "TypeScript"],
                 ["Redux", "Next.js", "Tailwind", "Jest", "Cypress", "Figma", "GraphQL", "Vue", "Node.js"]),
    "frontend_angular": (["Angular Developer", "UI Engineer", "Senior UI Engineer", "UI Architect"],
                         ["Angular", "TypeScript", "HTML/CSS", "JavaScript"],
                         ["Jest", "REST APIs", "Node.js", "Agile", "Figma"]),
    "backend_python": (["Python Developer", "Backend Engineer", "Senior Backend Engineer", "Principal Engineer"],
                       ["Python", "Django", "PostgreSQL", "REST APIs"],
                       ["FastAPI", "Flask", "Redis", "AWS", "Docker", "Kafka", "Celery", "Microservices", "System Design"]),
    "backend_java": (["Java Developer", "Software Engineer", "Senior Software Engineer", "Tech Lead"],
                     ["Java", "Spring Boot", "SQL", "Microservices"],
                     ["Kafka", "AWS", "Docker", "Kubernetes", "MySQL", "REST APIs", "System Design"]),
    "devops": (["DevOps Engineer", "Site Reliability Engineer", "Senior DevOps Engineer", "Platform Lead"],
               ["Docker", "Kubernetes", "AWS", "Terraform"],
               ["CI/CD", "Linux", "Python", "Go", "GCP", "Azure"]),
    "data_science": (["Data Analyst", "Data Scientist", "Senior Data Scientist", "Lead Data Scientist"],
                     ["Python", "Machine Learning", "Pandas", "SQL"],
                     ["scikit-learn", "TensorFlow", "PyTorch", "Spark", "Tableau", "NLP", "Airflow", "LLMs"]),
    "mobile": (["Mobile Developer", "Mobile Engineer", "Senior Mobile Engineer", "Mobile Lead"],
               ["React Native", "JavaScript", "TypeScript"],
               ["Android", "iOS", "Kotlin", "Swift", "Flutter", "Redux", "Jest", "React"]),
    "dotnet": ([".NET Developer", "Software Engineer", "Senior .NET Engineer", "Solutions Architect"],
               ["C#", ".NET", "SQL"],
               ["Azure", "Angular", "React", "Microservices", "Docker"]),
    "qa": (["QA Engineer", "SDET", "Senior SDET", "QA Lead"],
           ["Selenium", "Python", "Unit Testing"],
           ["Cypress", "Jest", "JavaScript", "CI/CD", "Agile", "Java"]),
}
ARCH_WEIGHTS = {"fullstack_js": 18, "frontend": 16, "frontend_angular": 7, "backend_python": 12, "backend_java": 10,
                "devops": 8, "data_science": 10, "mobile": 7, "dotnet": 6, "qa": 4}

VERBS = ["Built", "Designed", "Led development of", "Implemented", "Migrated", "Optimized", "Shipped",
         "Architected", "Maintained", "Refactored"]
THINGS = ["a customer-facing dashboard", "the payments service", "an internal admin portal", "a real-time notifications system",
          "the checkout flow", "a reporting pipeline", "the onboarding experience", "a design-system component library",
          "the search API", "a multi-tenant SaaS platform", "the mobile companion app", "an analytics platform"]
IMPACT = ["reducing page load time by {n}%", "cutting infrastructure cost by {n}%", "serving {m}M monthly users",
          "improving conversion by {n}%", "reducing incident rate by {n}%", "lowering p95 latency by {n}%"]
SOFT = ["Mentored {k} junior engineers", "Collaborated with product and design in an Agile team",
        "Ran code reviews and set engineering standards", "Partnered with stakeholders to define the roadmap"]


def _bullet(rng: random.Random, skills: list[str], quantified: bool) -> str:
    s = rng.sample(skills, k=min(len(skills), rng.choice([1, 2, 2, 3])))
    b = f"{rng.choice(VERBS)} {rng.choice(THINGS)} using {', '.join(s)}"
    if quantified:
        b += ", " + rng.choice(IMPACT).format(n=rng.randint(12, 65), m=rng.randint(1, 20))
    return b


def make_resume(idx: int, rng: random.Random, used_names: set[str]) -> str:
    arch = rng.choices(list(ARCHETYPES), weights=list(ARCH_WEIGHTS.values()))[0]
    titles, core, secondary = ARCHETYPES[arch]
    while True:
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if name not in used_names:
            used_names.add(name)
            break
    years = rng.choice([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 14])
    seniority = 0 if years < 2 else 1 if years < 5 else 2 if years < 9 else 3
    title = titles[seniority]

    my_core = [s for s in core if rng.random() < 0.88]
    my_secondary = rng.sample(secondary, k=rng.randint(1, min(6, len(secondary))))
    skills = list(dict.fromkeys(my_core + my_secondary + (["Agile"] if rng.random() < 0.5 else [])))
    quantifier = rng.random()  # how much this person quantifies impact

    # work history, most recent first
    roles_txt = []
    remaining, end = max(years, 0.5), CURRENT_YEAR
    role_titles = titles[: seniority + 1][::-1]
    n_roles = 1 if years <= 2 else rng.randint(2, min(4, 1 + years // 2))
    for r in range(n_roles):
        span = remaining if r == n_roles - 1 else max(1, round(remaining / (n_roles - r) + rng.uniform(-1, 1)))
        span = int(max(1, min(span, remaining)))
        start = end - span
        # older roles use a subset of skills (newer tech picked up later)
        role_skills = skills if r == 0 else rng.sample(skills, k=max(1, int(len(skills) * rng.uniform(0.4, 0.8))))
        rt = role_titles[min(r, len(role_titles) - 1)]
        end_txt = "Present" if r == 0 else str(end)
        bullets = [_bullet(rng, role_skills, rng.random() < quantifier) for _ in range(rng.randint(2, 4))]
        if seniority >= 2 and r == 0 and rng.random() < 0.7:
            bullets.append(rng.choice(SOFT).format(k=rng.randint(2, 6)))
        roles_txt.append(f"{rt}, {rng.choice(COMPANIES)} ({start} - {end_txt})\n" + "\n".join(f"- {b}" for b in bullets))
        end = start - (1 if rng.random() < 0.12 else 0)  # occasional career gap
        remaining -= span
        if remaining <= 0:
            break

    edu = rng.choice(DEGREES) if rng.random() < 0.93 else ""
    certs = [CERTS[s] for s in skills if s in CERTS and rng.random() < 0.35]
    summary_skills = ", ".join(skills[:4])
    summary = (f"{title} with {years} year{'s' if years != 1 else ''} of experience working with {summary_skills}. "
               f"{rng.choice(['Passionate about clean, maintainable code.', 'Focused on shipping user-centric products.', 'Enjoys solving performance and scale problems.', 'Comfortable owning features end to end.'])}")
    email = name.lower().replace(" ", ".") + "@example.com"
    parts = [f"Name: {name}", f"Email: {email}", f"Title: {title}", f"Location: {rng.choice(CITIES)}",
             f"Total Experience: {years} years", "", "SUMMARY", summary, "", "SKILLS", ", ".join(skills), "",
             "EXPERIENCE", "\n\n".join(roles_txt), "", "EDUCATION", edu or "-"]
    if certs:
        parts += ["", "CERTIFICATIONS", *certs]
    return "\n".join(parts) + "\n"


JOHN = """Name: John Smith
Email: john.smith@example.com
Title: Senior Frontend Engineer
Location: Chennai
Total Experience: 7 years

SUMMARY
Senior Frontend Engineer with 7 years of experience building React and TypeScript applications at scale. Strong in performance tuning and design systems.

SKILLS
React, TypeScript, JavaScript, Redux, Next.js, Node.js, REST APIs, Jest, GraphQL, HTML/CSS, Agile

EXPERIENCE
Senior Frontend Engineer, ShopSphere (2022 - Present)
- Led development of the checkout flow using React, TypeScript and Next.js, improving conversion by 18%
- Built a GraphQL gateway and REST APIs in Node.js serving 6M monthly users
- Mentored 4 junior engineers and ran code reviews

Frontend Engineer, Globex (2019 - 2022)
- Built a design-system component library using React, Redux and Jest, reducing page load time by 35%
- Implemented the onboarding experience using TypeScript and React

Frontend Developer, Initech (2019 - 2019)
- Maintained an internal admin portal using JavaScript and HTML/CSS

EDUCATION
B.Tech Computer Science, Anna University
"""

JANE = """Name: Jane Doe
Email: jane.doe@example.com
Title: Full Stack Engineer
Location: Bengaluru
Total Experience: 4 years

SUMMARY
Full Stack Engineer with 4 years of experience shipping cloud-native products on AWS with React and Node.js. Strong DevOps background.

SKILLS
React, Node.js, JavaScript, AWS, Docker, Kubernetes, Terraform, PostgreSQL, CI/CD, Express

EXPERIENCE
Full Stack Engineer, PayStream (2024 - Present)
- Architected a multi-tenant SaaS platform using Node.js, AWS and Kubernetes, cutting infrastructure cost by 40%
- Built a customer-facing dashboard using React and Express
- Set up CI/CD and Terraform pipelines for 12 services

Full Stack Developer, HealthBridge (2022 - 2024)
- Implemented the payments service using Node.js, PostgreSQL and Docker
- Shipped the onboarding experience using React and JavaScript

EDUCATION
B.E. Information Technology, VIT

CERTIFICATIONS
AWS Certified Solutions Architect – Associate
"""

JD = """Senior Full Stack Engineer — FinEdge Payments

About the role
We are looking for a Senior Full Stack Engineer to build our merchant dashboard and payments APIs.

Must have
- 5+ years of professional software engineering experience
- Strong React experience (3+ years)
- TypeScript
- Node.js for backend services
- Experience designing REST APIs

Nice to have
- AWS
- Docker
- GraphQL
- Experience with PostgreSQL
- Kubernetes is a plus

Education
- Bachelor's degree in Computer Science or related field preferred
"""


def main(n: int = 100, seed: int = 42) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    JD_DIR.mkdir(parents=True, exist_ok=True)
    for old in RES_DIR.glob("*.txt"):
        old.unlink()
    rng = random.Random(seed)
    used = {"John Smith", "Jane Doe"}
    (RES_DIR / "cand_001.txt").write_text(JOHN, encoding="utf-8")
    (RES_DIR / "cand_002.txt").write_text(JANE, encoding="utf-8")
    for i in range(3, n + 1):
        (RES_DIR / f"cand_{i:03d}.txt").write_text(make_resume(i, rng, used), encoding="utf-8")
    (JD_DIR / "senior_fullstack.txt").write_text(JD, encoding="utf-8")
    print(f"Wrote {n} resumes to {RES_DIR} and a JD to {JD_DIR}")


if __name__ == "__main__":
    main()
