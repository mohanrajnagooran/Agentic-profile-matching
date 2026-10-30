"""
Generate realistic PDF/DOCX resumes in different real-world layouts (for testing uploads).

    pip install reportlab python-docx
    python tools/make_sample_uploads.py      # → data/sample_uploads/

Only needed to regenerate the samples; the generated files are already in the repo.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_uploads"

# Layout 1: classic one-column, "Title | Company | Mon YYYY – Present", grouped skills
ARJUN = {
    "file": "Arjun_Mehta_Resume.pdf",
    "lines": [
        ("h1", "ARJUN MEHTA"),
        ("small", "Senior Full Stack Engineer  |  arjun.mehta@gmail.com  |  +91 98400 12345  |  linkedin.com/in/arjunmehta"),
        ("h2", "PROFESSIONAL SUMMARY"),
        ("p", "Full stack engineer with 8+ years of experience building payment and commerce platforms with React, "
              "TypeScript and Node.js. Comfortable owning features from API design to production on AWS."),
        ("h2", "TECHNICAL SKILLS"),
        ("p", "Languages: TypeScript, JavaScript, Python, SQL"),
        ("p", "Frontend: React, Next.js, Redux, Tailwind"),
        ("p", "Backend: Node.js, Express, REST APIs, GraphQL"),
        ("p", "Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD   |   Databases: PostgreSQL, Redis"),
        ("h2", "WORK EXPERIENCE"),
        ("b", "Senior Software Engineer | Razorpay | Jan 2022 – Present"),
        ("li", "Led the rebuild of the merchant dashboard in React and TypeScript, improving page load time by 42%"),
        ("li", "Designed REST APIs in Node.js for refunds and settlements handling 3M requests per day"),
        ("li", "Mentored 3 engineers and introduced contract testing in the CI/CD pipeline"),
        ("b", "Software Engineer | Freshworks | Jul 2019 – Dec 2021"),
        ("li", "Built a GraphQL gateway in Node.js and PostgreSQL used by 12 product teams"),
        ("li", "Migrated services to Docker and Kubernetes on AWS, cutting infrastructure cost by 25%"),
        ("b", "Associate Engineer | Zoho | Jun 2018 – Jun 2019"),
        ("li", "Developed internal tools with JavaScript and Python"),
        ("h2", "EDUCATION"),
        ("p", "B.Tech in Computer Science and Engineering, SRM Institute of Science and Technology — 2018"),
        ("h2", "CERTIFICATIONS"),
        ("p", "AWS Certified Developer – Associate"),
    ],
}

# Layout 2: title on one line, "Company, City        MM/YYYY - MM/YYYY" below, ● bullets, skills at the bottom
PRIYA = {
    "file": "Priya_Raman_CV.pdf",
    "lines": [
        ("h1", "Priya Raman"),
        ("small", "Chennai, India · priya.raman@outlook.com · github.com/priyaraman"),
        ("h2", "Profile"),
        ("p", "Frontend-focused engineer who enjoys design systems, accessibility and performance work."),
        ("h2", "Experience"),
        ("b", "Frontend Engineer"),
        ("p", "Swiggy, Bengaluru          04/2023 - Present"),
        ("li", "Built the restaurant partner app in React and TypeScript with Redux Toolkit"),
        ("li", "Created a component library with Jest and Cypress tests, used across 9 squads"),
        ("b", "UI Developer"),
        ("p", "Tata Consultancy Services, Chennai          08/2020 - 03/2023"),
        ("li", "Developed Angular and JavaScript screens for a banking client; wrote REST API integrations"),
        ("li", "Reduced bundle size by 30% through lazy loading"),
        ("h2", "Education"),
        ("p", "B.E. Electronics and Communication, Anna University (2020)"),
        ("h2", "Skills"),
        ("p", "React, TypeScript, JavaScript, Angular, HTML/CSS, Redux, Jest, Cypress, Figma, REST APIs"),
    ],
}

# Layout 3: fresher — objective, projects, internship, no years in summary
KARAN = {
    "file": "Karan_Shah_Fresher.pdf",
    "lines": [
        ("h1", "KARAN SHAH"),
        ("small", "karan.shah2003@gmail.com | +91 90000 11111 | Pune"),
        ("h2", "OBJECTIVE"),
        ("p", "Final-year computer science student looking for a backend developer role."),
        ("h2", "EDUCATION"),
        ("p", "B.Tech Computer Engineering, VIT Pune — CGPA 8.6 (2022 – 2026)"),
        ("h2", "INTERNSHIPS"),
        ("b", "Backend Intern — Postman, Bengaluru (May 2025 – Jul 2025)"),
        ("li", "Built REST APIs in Python with FastAPI and PostgreSQL for an internal analytics service"),
        ("li", "Containerised the service with Docker"),
        ("h2", "PROJECTS"),
        ("li", "Expense tracker: React frontend with a Node.js and MongoDB backend"),
        ("li", "Movie recommender using Python, Pandas and scikit-learn"),
        ("h2", "SKILLS"),
        ("p", "Python, FastAPI, Node.js, React, PostgreSQL, MongoDB, Docker, Git"),
    ],
}

# Layout 4 (DOCX): "Title at Company" + "Mar 2017 to Feb 2024", headings in Title Case with colons
MEERA = {
    "file": "Meera_Iyer_Resume.docx",
    "lines": [
        ("h1", "Meera Iyer"),
        ("small", "meera.iyer@example.com | Hyderabad"),
        ("h2", "Summary:"),
        ("p", "Backend and DevOps engineer with 9 years of experience in Java, Spring Boot and AWS."),
        ("h2", "Core Competencies:"),
        ("p", "Java, Spring Boot, Microservices, Kafka, AWS, Docker, Kubernetes, Terraform, PostgreSQL"),
        ("h2", "Professional Experience:"),
        ("b", "Lead Engineer at Flipkart"),
        ("p", "Mar 2020 to Present"),
        ("li", "Architected Kafka-based order pipeline on Kubernetes processing 20M events per day"),
        ("li", "Managed a team of 6 engineers; owned Terraform modules for AWS"),
        ("b", "Software Engineer at Infosys"),
        ("p", "Jun 2015 to Feb 2020"),
        ("li", "Built Spring Boot microservices with PostgreSQL for an insurance platform"),
        ("h2", "Education:"),
        ("p", "M.Tech Software Engineering, BITS Pilani"),
    ],
}


def make_pdf(spec: dict) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

    ss = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=4),
        "small": ParagraphStyle("small", parent=ss["Normal"], fontSize=9, alignment=1, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4),
        "b": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica-Bold", spaceBefore=6),
        "p": ss["Normal"],
    }
    story, bullets = [], []

    def flush():
        if bullets:
            story.append(ListFlowable([ListItem(Paragraph(t, ss["Normal"])) for t in bullets], bulletType="bullet", start="•"))
            bullets.clear()

    for kind, text in spec["lines"]:
        if kind == "li":
            bullets.append(text)
            continue
        flush()
        story.append(Paragraph(text.replace("&", "&amp;"), styles[kind]))
    flush()
    story.append(Spacer(1, 6))
    SimpleDocTemplate(str(OUT / spec["file"]), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=40).build(story)


def make_docx(spec: dict) -> None:
    import docx

    d = docx.Document()
    for kind, text in spec["lines"]:
        if kind == "h1":
            d.add_heading(text, level=0)
        elif kind == "h2":
            d.add_heading(text, level=2)
        elif kind == "li":
            d.add_paragraph(text, style="List Bullet")
        elif kind == "b":
            d.add_paragraph().add_run(text).bold = True
        else:
            d.add_paragraph(text)
    d.save(str(OUT / spec["file"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in (ARJUN, PRIYA, KARAN):
        make_pdf(spec)
    make_docx(MEERA)
    print(f"Wrote sample uploads to {OUT}")


if __name__ == "__main__":
    main()
