"""Generate nine fictional, realistic resumes (PDF) for trying the app by hand.

Unlike the unit-test fixtures these are deliberately messy in the ways real resumes are:
multi-page, sidebars with skill bars and a photo placeholder, a full-width header band, a
timeline table with dates in their own column, text flowing across two columns, running
headers and footers, promotions inside one company, overlapping jobs, a career break, tables
of skills, publications, certifications, a bilingual degree name, and one scanned
(image-only) two-column page.

Everyone, every company and every contact detail is invented (emails use example.com).

Usage: python scripts/make_realistic_resumes.py [output_dir]      (default: local_resumes)
"""

from __future__ import annotations

import html as htmllib
import io
import sys
from pathlib import Path

import pymupdf

A4 = pymupdf.paper_rect("a4")  # 595 x 842 points
DEFAULT_OUT = "local_resumes"
NAVY, TEAL, GREY, PLUM = (0.11, 0.2, 0.36), (0.07, 0.35, 0.38), (0.94, 0.94, 0.95), (0.3, 0.12, 0.35)
warnings: list[str] = []


def esc(text: str) -> str:
    return htmllib.escape(text, quote=False)


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def jobs_html(jobs: list[dict], date_style: str = "table") -> str:
    """Experience as HTML. A job may hold several roles (a promotion inside one company)."""
    out = []
    for job in jobs:
        head = f"<b>{esc(job['company'])}</b>, {esc(job['place'])}"
        if date_style == "table":
            out.append(f'<table style="width:100%"><tr><td>{head}</td>'
                       f'<td style="text-align:right;width:120pt">{esc(job["dates"])}</td></tr></table>')
        else:
            out.append(f"<p>{head}<br>{esc(job['dates'])}</p>")
        for role in job["roles"]:
            title = f"<p><i>{esc(role['title'])}</i>"
            title += f" ({esc(role['dates'])})</p>" if role.get("dates") else "</p>"
            out.append(title)
            out.append(bullets(role["bullets"]))
    return "".join(out)


# --- layout helpers ---------------------------------------------------------------------


def flow(html: str, css: str, rects_per_page: list[list[pymupdf.Rect]]) -> bytes:
    """Let a Story flow through the given rectangles, page after page."""
    story = pymupdf.Story(html=html, user_css=css)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more, page_no = 1, 0
    while more:
        dev = writer.begin_page(A4)
        for rect in rects_per_page[min(page_no, len(rects_per_page) - 1)]:
            more, _ = story.place(rect)
            story.draw(dev)
            if not more:
                break
        writer.end_page()
        page_no += 1
    writer.close()
    return buffer.getvalue()


def margins(left=48, top=46, right=48, bottom=62) -> pymupdf.Rect:
    return pymupdf.Rect(left, top, A4.width - right, A4.height - bottom)


def box(page, rect, html, css, label=""):
    """Write HTML into a fixed rectangle; remember if the text had to shrink to fit."""
    spare, scale = page.insert_htmlbox(pymupdf.Rect(rect), html, css=css)
    if scale < 0.9 or spare < 0:
        warnings.append(f"{label or 'box'}: text shrank to {scale:.0%} (spare {spare:.0f}pt)")


def finish(data: bytes, title: str, author: str, post=None) -> bytes:
    doc = pymupdf.open(stream=data, filetype="pdf")
    if post:
        post(doc)
    doc.set_metadata({"title": title, "author": author, "producer": "Resume Builder Pro"})
    out = doc.tobytes(deflate=True)
    doc.close()
    return out


def rasterise(data: bytes, dpi: int = 150) -> bytes:
    """Turn a PDF into an image-only one, like a scan from a copier."""
    src = pymupdf.open(stream=data, filetype="pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    result = out.tobytes()
    src.close()
    out.close()
    return result


def footer(text: str, from_page: int = 0, running_header: str | None = None):
    def post(doc):
        n = doc.page_count
        for i, page in enumerate(doc):
            page.insert_text((48, A4.height - 30), f"{text}  |  Page {i + 1} of {n}",
                             fontsize=8, color=(0.45, 0.45, 0.45))
            if running_header and i >= from_page:
                page.insert_text((48, 28), running_header, fontsize=8, color=(0.45, 0.45, 0.45))
    return post


def skill_rows(page, x, y, width, skills, text_color, track, fill, step=32):
    """Skill names with a drawn level bar under each (the bar is graphics, not text)."""
    css = f"body {{ font-family: sans-serif; font-size: 9.6pt; color: {text_color}; }} " \
          "p { margin: 0; }"
    for name, level in skills:
        box(page, (x, y, x + width, y + 20), f"<p>{esc(name)}</p>", css, f"skill {name}")
        page.draw_rect(pymupdf.Rect(x, y + 21, x + width, y + 25), color=None, fill=track)
        page.draw_rect(pymupdf.Rect(x, y + 21, x + width * level, y + 25), color=None, fill=fill)
        y += step
    return y


BASE_CSS = """
body { font-family: sans-serif; font-size: 10.5pt; color: #222222; }
h1 { font-size: 23pt; margin: 0; color: #1c3354; }
h2 { font-size: 12pt; margin-top: 12pt; margin-bottom: 3pt; color: #1c3354; }
p { margin-top: 2pt; margin-bottom: 2pt; }
ul { margin-top: 1pt; margin-bottom: 5pt; }
li { margin-bottom: 1.5pt; }
td { padding-top: 2pt; }
"""


# --- 1. Ananya Krishnan: classic two-page CV, promotions, running footer ---------------


def resume_classic() -> bytes:
    jobs = [
        {"company": "Zenith Cloud Systems", "place": "Bengaluru", "dates": "March 2021 - Present",
         "roles": [
             {"title": "Staff Platform Engineer", "dates": "April 2023 - Present", "bullets": [
                 "Own the internal developer platform used by 140 engineers across 22 teams; cut median time from merge to production from 47 minutes to 9 minutes.",
                 "Designed a multi-region Kubernetes fleet on AWS (EKS) and GCP (GKE) with Terraform and ArgoCD, reaching 99.98% availability for tier-1 services.",
                 "Introduced a cost dashboard built on Prometheus and OpenCost that exposed 31% idle capacity; rightsizing saved USD 1.2M per year.",
                 "Mentor six engineers and lead the quarterly production-readiness review programme.",
             ]},
             {"title": "Senior Platform Engineer", "dates": "March 2021 - March 2023", "bullets": [
                 "Replaced a hand-rolled deployment tool with GitHub Actions and Helm; adoption reached 100% of services in five months.",
                 "Built OpenTelemetry tracing for 60 Go and Python services, shortening average incident diagnosis from 55 to 20 minutes.",
                 "Led the migration of 38 stateful workloads from VM images to StatefulSets with zero customer-visible downtime.",
             ]},
         ]},
        {"company": "Nimbus Retail Technologies", "place": "Pune", "dates": "July 2017 - February 2021",
         "roles": [
             {"title": "Software Engineer II", "dates": "June 2019 - February 2021", "bullets": [
                 "Re-architected the checkout service from a Django monolith into four Go microservices, raising peak throughput from 400 to 3,500 orders per minute.",
                 "Implemented blue-green deployments on Kubernetes and automated rollback on error-budget burn.",
                 "Reduced PostgreSQL p99 latency by 62% through partitioning, read replicas and query plan reviews.",
             ]},
             {"title": "Software Engineer", "dates": "July 2017 - May 2019", "bullets": [
                 "Developed inventory synchronisation APIs in Python and Flask consumed by 1,800 stores.",
                 "Wrote the team's first CI pipeline in Jenkins; test runtime fell from 38 to 11 minutes after parallelisation.",
             ]},
         ]},
        {"company": "Orchid Payments", "place": "Hyderabad", "dates": "June 2015 - June 2017",
         "roles": [{"title": "Associate Software Engineer", "bullets": [
             "Built reconciliation jobs in Java and Spring Boot that processed 2.4 million card transactions per day.",
             "Maintained Oracle and MySQL schemas for the settlement module; authored 120 regression tests.",
         ]}]},
    ]
    html = f"""
    <h1>Ananya Krishnan</h1>
    <p style="font-size:12pt;color:#555555">Staff Platform Engineer</p>
    <p>Bengaluru, India | ananya.krishnan@example.com | +91 98450 01234 | linkedin.com/in/ananya-krishnan-example | github.com/ananyak-example</p>
    <h2>PROFESSIONAL SUMMARY</h2>
    <p>Platform engineer with nine years of experience building reliable, cost-aware cloud infrastructure for high-traffic commerce and payments systems. Strong in Kubernetes, Terraform and observability, with a record of turning ad-hoc operations into self-service platforms. Comfortable leading cross-team migrations and mentoring.</p>
    <h2>PROFESSIONAL EXPERIENCE</h2>
    {jobs_html(jobs)}
    <h2>EDUCATION</h2>
    <table style="width:100%"><tr><td><b>B.E. in Computer Science and Engineering</b>, PES Institute of Technology, Bengaluru</td><td style="text-align:right;width:120pt">2011 - 2015</td></tr></table>
    <h2>TECHNICAL SKILLS</h2>
    <p><b>Languages:</b> Go, Python, Bash, SQL, Java<br>
    <b>Cloud and containers:</b> AWS, GCP, Kubernetes, Docker, Helm<br>
    <b>Infrastructure as code and delivery:</b> Terraform, ArgoCD, GitHub Actions, Jenkins<br>
    <b>Observability:</b> Prometheus, Grafana, OpenTelemetry, Loki<br>
    <b>Data stores:</b> PostgreSQL, Redis, Kafka</p>
    <h2>CERTIFICATIONS</h2>
    <ul><li>Certified Kubernetes Administrator (CKA), 2022</li>
    <li>AWS Certified Solutions Architect - Professional, 2023</li>
    <li>HashiCorp Certified: Terraform Associate, 2021</li></ul>
    """
    data = flow(html, BASE_CSS, [[margins()]])
    return finish(data, "Ananya Krishnan - Resume", "Ananya Krishnan",
                  footer("Ananya Krishnan - Staff Platform Engineer"))


# --- 2. Daniel Okafor: dark left sidebar, photo, skill bars, projects, publications ----


def resume_sidebar(
    name="Daniel Okafor", title="Senior Machine Learning Engineer", accent=NAVY, scan=False
) -> bytes:
    doc = pymupdf.open()
    side_css = "body { font-family: sans-serif; font-size: 9.6pt; color: #ffffff; } " \
               "h3 { font-size: 10pt; margin: 0 0 4pt 0; color: #9fc2ff; } p { margin: 2pt 0; } " \
               "ul { margin: 0; } li { margin-bottom: 2pt; }"
    main_css = BASE_CSS + "h1 { font-size: 24pt; }"
    skills = [("Python", 0.95), ("PyTorch", 0.9), ("SQL", 0.85), ("Spark", 0.75),
              ("MLOps / Kubeflow", 0.7), ("Docker and Kubernetes", 0.7), ("AWS SageMaker", 0.65)]

    # page 1
    p1 = doc.new_page(width=A4.width, height=A4.height)
    p1.draw_rect(pymupdf.Rect(0, 0, 190, A4.height), color=None, fill=accent)
    p1.draw_circle(pymupdf.Point(95, 78), 44, color=(1, 1, 1), fill=(0.72, 0.74, 0.78), width=2)
    box(p1, (16, 140, 176, 270), """
        <h3>CONTACT</h3>
        <p>London, United Kingdom</p><p>daniel.okafor@example.com</p>
        <p>+44 7700 900123</p><p>linkedin.com/in/daniel-okafor-example</p>
        <p>github.com/dokafor-example</p>""", side_css, "contact")
    box(p1, (16, 282, 176, 300), "<h3>SKILLS</h3>", side_css)
    y = skill_rows(p1, 16, 302, 158, skills, "#ffffff", (0.28, 0.35, 0.5), (0.62, 0.76, 1.0))
    box(p1, (16, y + 8, 176, y + 120), """
        <h3>LANGUAGES</h3><p>English (native)</p><p>Yoruba (fluent)</p><p>French (basic)</p>""",
        side_css, "languages")

    main1 = f"""
    <h1>{name}</h1>
    <p style="font-size:12pt;color:#555555">{title}</p>
    <h2>PROFILE</h2>
    <p>Machine learning engineer with seven years of experience taking ranking, forecasting and fraud models from notebooks to production at scale. Focused on measurable business impact, reproducible pipelines and monitoring that catches drift before customers do.</p>
    <h2>EXPERIENCE</h2>
    {jobs_html([
        {"company": "Helix Analytics", "place": "London", "dates": "Jan 2022 - Present", "roles": [
            {"title": "Senior Machine Learning Engineer", "bullets": [
                "Lead a team of five building a real-time fraud scoring service (Python, PyTorch, Kafka) that screens 9 million payments a day at a median latency of 38 ms.",
                "Cut false positives by 24% while holding recall steady, saving the fraud-ops team roughly 600 review hours a month.",
                "Introduced feature stores and shadow deployments; model release time dropped from three weeks to two days.",
                "Built drift monitoring on Prometheus and Evidently with automatic rollback.",
            ]}]},
        {"company": "Brightpath AI", "place": "Lagos (remote)", "dates": "Feb 2019 - Dec 2021", "roles": [
            {"title": "Machine Learning Engineer", "bullets": [
                "Developed a product-ranking model for a marketplace with 4 million monthly users; click-through rose 11% in A/B tests.",
                "Created the company's first batch inference platform on Spark and Airflow, processing 300 GB nightly.",
                "Mentored two junior engineers and presented quarterly results to the executive team.",
            ]}]},
    ], date_style="table")}
    """
    box(p1, (206, 38, 560, 810), main1, main_css, "sidebar main p1")

    # page 2
    p2 = doc.new_page(width=A4.width, height=A4.height)
    p2.draw_rect(pymupdf.Rect(0, 0, 190, A4.height), color=None, fill=accent)
    box(p2, (16, 40, 176, 420), """
        <h3>CERTIFICATIONS</h3>
        <ul><li>TensorFlow Developer Certificate, 2020</li>
        <li>Databricks Certified Machine Learning Associate, 2023</li>
        <li>AWS Certified Machine Learning - Specialty, 2024</li></ul><br>
        <h3>AWARDS</h3>
        <ul><li>Helix Innovation Award 2023</li><li>Best Paper Runner-up, Applied ML Workshop 2021</li></ul>""",
        side_css, "side p2")
    main2 = f"""
    {jobs_html([{"company": "Savanna Telecom", "place": "Lagos", "dates": "Aug 2017 - Jan 2019", "roles": [
        {"title": "Data Analyst", "bullets": [
            "Automated churn reporting in Python and SQL, replacing a 3-day spreadsheet process with a daily dashboard.",
            "Built a churn-propensity model with scikit-learn that guided retention offers to 80,000 customers.",
        ]}]}])}
    <h2>SELECTED PROJECTS</h2>
    <p><b>Open-source: calibrate-rank</b> - a Python library for probability calibration of ranking models (1.4k GitHub stars, used by three public Kaggle winners).</p>
    <p><b>Fraud simulator</b> - synthetic payment-stream generator used to load-test scoring services before release.</p>
    <h2>PUBLICATIONS</h2>
    <ul><li>Okafor, D. and Chen, L. "Calibrated ranking under sparse feedback." Proceedings of the Applied ML Workshop, 2021.</li>
    <li>Okafor, D. "Shadow deployment patterns for fraud models." Helix Engineering Blog, 2023.</li></ul>
    <h2>EDUCATION</h2>
    <table style="width:100%"><tr><td><b>MSc in Data Science</b>, University of Edinburgh</td><td style="text-align:right;width:120pt">2016 - 2017</td></tr>
    <tr><td><b>BSc in Statistics</b>, University of Ibadan</td><td style="text-align:right;width:120pt">2011 - 2015</td></tr></table>
    """
    box(p2, (206, 38, 560, 810), main2, main_css, "sidebar main p2")
    data = doc.tobytes()
    doc.close()
    data = finish(data, f"{name} - CV", name)
    return rasterise(data) if scan else data


# --- 3. Mei Tanaka: academic CV, education first, years only, skills table -------------


def resume_academic() -> bytes:
    css = BASE_CSS + "h1 { font-size: 21pt; color: #222222; } h2 { color: #5a2a82; }"
    html = f"""
    <h1>Mei Tanaka, PhD</h1>
    <p>Senior Data Scientist | Seattle, Washington</p>
    <p>mei.tanaka@example.com | +1 206 555 0147 | scholar.example.com/mei-tanaka</p>
    <h2>Research Interests</h2>
    <p>Causal inference, experimentation platforms, Bayesian hierarchical models and their use in healthcare operations and marketplace pricing.</p>
    <h2>Education</h2>
    <table style="width:100%">
    <tr><td><b>PhD, Statistics</b>, University of Washington</td><td style="text-align:right;width:120pt">2015 - 2020</td></tr>
    <tr><td>Dissertation: Doubly robust estimation for staggered rollouts in clustered trials</td><td></td></tr>
    <tr><td><b>BSc, Mathematics</b> (Honours), Waseda University, Tokyo</td><td style="text-align:right;width:120pt">2011 - 2015</td></tr>
    </table>
    <h2>Professional Background</h2>
    {jobs_html([
        {"company": "Redwood Marketplace", "place": "Seattle", "dates": "2023 - Present", "roles": [
            {"title": "Senior Data Scientist, Pricing Science", "bullets": [
                "Own the experimentation methodology for a marketplace with 11 million listings; designed switchback and geo-experiment frameworks now used by 14 product teams.",
                "Built a Bayesian price-elasticity model (Stan, Python) that raised take-rate by 1.8 points without hurting conversion.",
                "Established the variance-reduction library (CUPED and regression adjustment) that shortened median test duration from 21 to 12 days.",
            ]}]},
        {"company": "Cascade Health Analytics", "place": "Seattle", "dates": "2020 - 2023", "roles": [
            {"title": "Data Scientist", "bullets": [
                "Forecast emergency-department demand for 9 hospitals; mean absolute error fell 19% against the incumbent model.",
                "Evaluated a patient-flow intervention with difference-in-differences and synthetic control, informing a USD 4M staffing decision.",
                "Mentored interns and ran the internal reading group on causal machine learning.",
            ]}]},
        {"company": "Allen Institute for Computational Biology", "place": "Seattle", "dates": "2019 (summer)", "roles": [
            {"title": "Research Intern", "bullets": [
                "Implemented Hamiltonian Monte Carlo samplers for single-cell expression models.",
            ]}]},
    ], date_style="table")}
    <h2>Technical Proficiencies</h2>
    <table style="width:100%">
    <tr><td><b>Statistics</b></td><td>Causal inference, Bayesian modelling, mixed-effects models, survival analysis, A/B testing</td></tr>
    <tr><td><b>Languages</b></td><td>Python, R, SQL, Stan</td></tr>
    <tr><td><b>Libraries</b></td><td>pandas, scikit-learn, statsmodels, PyMC, XGBoost</td></tr>
    <tr><td><b>Data platforms</b></td><td>Snowflake, BigQuery, Airflow, dbt</td></tr>
    </table>
    <h2>Selected Publications</h2>
    <ol>
    <li>Tanaka, M. and Whitfield, R. (2022). Variance reduction for marketplace experiments with interference. Journal of Causal Inference, 10(2), 211-236.</li>
    <li>Tanaka, M. (2020). Doubly robust estimation for staggered rollouts. Annals of Applied Statistics, 14(4), 1802-1829.</li>
    <li>Tanaka, M., Silva, P. and Brown, K. (2019). Bayesian demand forecasting in hospital emergency departments. Health Care Management Science, 22, 645-660.</li>
    </ol>
    <h2>Honours and Awards</h2>
    <ul><li>Best Applied Paper, Conference on Digital Experimentation, 2022</li>
    <li>NSF Graduate Research Fellowship, 2016 - 2019</li>
    <li>Department of Statistics Teaching Award, University of Washington, 2018</li></ul>
    <h2>Languages</h2>
    <p>English (fluent), Japanese (native)</p>
    """
    data = flow(html, css, [[margins()]])
    return finish(data, "Mei Tanaka - CV", "Mei Tanaka", footer("Mei Tanaka, PhD - Curriculum Vitae"))


# --- 4. Lars Johansson: header band, right sidebar, overlapping jobs, career break -----


def resume_band() -> bytes:
    doc = pymupdf.open()
    css = BASE_CSS + "h2 { color: #135a62; }"
    side_css = "body { font-family: sans-serif; font-size: 9.6pt; color: #222222; } " \
               "h3 { font-size: 10pt; margin: 8pt 0 3pt 0; color: #135a62; } " \
               "p { margin: 2pt 0; } ul { margin: 0 0 4pt 0; } li { margin-bottom: 2pt; }"
    p1 = doc.new_page(width=A4.width, height=A4.height)
    p1.draw_rect(pymupdf.Rect(0, 0, A4.width, 124), color=None, fill=TEAL)
    box(p1, (40, 14, 400, 118), """<p style="font-size:26pt;color:#ffffff"><b>Lars Johansson</b></p>
        <p style="font-size:12pt;color:#cfe8ea">Principal Site Reliability Engineer</p>""",
        "body { font-family: sans-serif; } p { margin: 3pt 0; }", "band name")
    box(p1, (385, 22, 575, 112), """<p style="color:#ffffff;font-size:9.6pt">Stockholm, Sweden</p>
        <p style="color:#ffffff;font-size:9.6pt">lars.johansson@example.com</p>
        <p style="color:#ffffff;font-size:9.6pt">+46 70 555 01 22</p>
        <p style="color:#ffffff;font-size:9.6pt">github.com/larsj-example</p>""",
        "body { font-family: sans-serif; }", "band contact")
    p1.draw_rect(pymupdf.Rect(392, 124, A4.width, A4.height), color=None, fill=GREY)

    main1 = f"""
    <h2>SUMMARY</h2>
    <p>Reliability engineer with ten years across banking and streaming. Specialises in SLO-driven operations, incident management and capacity planning for systems serving tens of millions of users. Equally at home writing Go tooling and facilitating blameless post-mortems.</p>
    <h2>EXPERIENCE</h2>
    {jobs_html([
        {"company": "Northlight Streaming", "place": "Stockholm", "dates": "May 2020 - Present", "roles": [
            {"title": "Principal Site Reliability Engineer", "bullets": [
                "Define and enforce SLOs for 45 services; error-budget policy reduced customer-facing incidents by 41% in two years.",
                "Led the move from a single data centre to active-active across two regions, surviving a full region failure with 4 minutes of degraded service.",
                "Built autoscaling for the video-transcode fleet (Kubernetes, KEDA) that lowered compute spend by 28% during off-peak.",
                "Run the on-call programme for 30 engineers; page volume per engineer fell by half after alert clean-up.",
            ]}]},
        {"company": "Freelance Infrastructure Consultant (self-employed)", "place": "Remote", "dates": "Sep 2017 - Apr 2020", "roles": [
            {"title": "Independent consultant", "bullets": [
                "Delivered Terraform and Kubernetes platforms for four start-ups, from initial design to hand-over.",
                "Ran incident-response training for a logistics scale-up; mean time to recovery improved from 90 to 35 minutes.",
            ]}]},
    ])}
    """
    box(p1, (40, 136, 372, 812), main1, css, "band main p1")
    box(p1, (402, 136, 583, 812), """
        <h3>CORE SKILLS</h3>
        <ul><li>SLOs and error budgets</li><li>Incident command</li><li>Capacity planning</li>
        <li>Chaos engineering</li><li>Cost optimisation</li></ul>
        <h3>TECHNOLOGY</h3>
        <p><b>Languages:</b> Go, Python, Bash</p>
        <p><b>Platforms:</b> Kubernetes, AWS, GCP, Nomad</p>
        <p><b>Tooling:</b> Terraform, Ansible, Prometheus, Grafana, Loki, PagerDuty</p>
        <p><b>Data:</b> PostgreSQL, Kafka, Cassandra</p>
        <h3>CERTIFICATIONS</h3>
        <ul><li>Certified Kubernetes Administrator, 2020</li>
        <li>AWS Certified DevOps Engineer - Professional, 2021</li>
        <li>Google Professional Cloud Architect, 2022</li></ul>
        <h3>LANGUAGES</h3><p>Swedish (native)<br>English (fluent)<br>Norwegian (good)</p>""",
        side_css, "band side p1")

    p2 = doc.new_page(width=A4.width, height=A4.height)
    p2.draw_rect(pymupdf.Rect(392, 0, A4.width, A4.height), color=None, fill=GREY)
    main2 = f"""
    {jobs_html([
        {"company": "Fjord Bank", "place": "Oslo", "dates": "Aug 2014 - Dec 2018", "roles": [
            {"title": "Senior Site Reliability Engineer", "dates": "Jul 2016 - Dec 2018", "bullets": [
                "Moved 120 batch and online services from physical servers to a private OpenStack cloud with Ansible, retiring 300 machines.",
                "Created the bank's first game-day programme; 14 latent failure modes were found and fixed before audits.",
                "Co-authored the regulator-facing disaster-recovery plan, passed without findings.",
            ]},
            {"title": "Site Reliability Engineer", "dates": "Aug 2014 - Jun 2016", "bullets": [
                "Automated certificate rotation, removing a recurring cause of weekend outages.",
                "Introduced Prometheus and Grafana alongside the legacy Nagios setup.",
            ]},
        ]},
        {"company": "Telenor Digital", "place": "Oslo", "dates": "Jun 2013 - Sep 2013", "roles": [
            {"title": "Operations intern", "bullets": ["Scripted nightly capacity reports for the mobile messaging platform."]}]},
    ])}
    <h2>CAREER BREAK</h2>
    <p>Jan 2019 - Oct 2019: sabbatical. Travelled for nine months while contributing to open-source observability tooling (merged patches to two Prometheus exporters).</p>
    <h2>EDUCATION</h2>
    <table style="width:100%"><tr><td><b>MSc in Computer Science</b>, KTH Royal Institute of Technology, Stockholm</td><td style="text-align:right;width:120pt">2011 - 2013</td></tr>
    <tr><td><b>BSc in Information Technology</b>, Uppsala University</td><td style="text-align:right;width:120pt">2008 - 2011</td></tr></table>
    """
    box(p2, (40, 40, 372, 812), main2, css, "band main p2")
    box(p2, (402, 40, 583, 300), """
        <h3>SPEAKING</h3>
        <ul><li>SREcon EMEA 2023: "Region failure drills that don't hurt"</li>
        <li>KubeCon EU 2022: "Autoscaling media workloads"</li></ul>
        <h3>OPEN SOURCE</h3>
        <ul><li>Maintainer, slo-tools (Go)</li><li>Contributor, Prometheus exporters</li></ul>""",
        side_css, "band side p2")
    data = doc.tobytes()
    doc.close()
    return finish(data, "Lars Johansson - Resume", "Lars Johansson")


# --- 5. Camila Rojas: timeline table, dates in their own column ------------------------


def resume_timeline() -> bytes:
    doc = pymupdf.open()
    css = BASE_CSS + "td { padding-top: 5pt; padding-bottom: 5pt; }"
    page1 = doc.new_page(width=A4.width, height=A4.height)
    page1.draw_rect(pymupdf.Rect(0, 0, A4.width, 112), color=None, fill=PLUM)
    box(page1, (40, 12, 380, 106), """<p style="font-size:25pt;color:#ffffff"><b>Camila Rojas</b></p>
        <p style="font-size:12pt;color:#ead7f0">Full-Stack Developer</p>""",
        "body { font-family: sans-serif; } p { margin: 3pt 0; }", "timeline name")
    box(page1, (390, 18, 575, 106), """<p style="color:#ffffff;font-size:9.6pt">Bogota, Colombia (remote)</p>
        <p style="color:#ffffff;font-size:9.6pt">camila.rojas@example.com</p>
        <p style="color:#ffffff;font-size:9.6pt">+57 300 555 0188</p>
        <p style="color:#ffffff;font-size:9.6pt">camilarojas.example.dev</p>""",
        "body { font-family: sans-serif; }", "timeline contact")
    page1.draw_line(pymupdf.Point(138, 196), pymupdf.Point(138, 425), color=(0.7, 0.55, 0.78), width=1.4)

    page1_html = """
    <p><b>ABOUT ME</b></p>
    <p>Full-stack developer with five years of experience shipping customer-facing web products for fintech and creative agencies. I like turning ambiguous requirements into small, tested releases, and I care about accessibility and performance budgets.</p>
    """
    box(page1, (40, 124, 560, 190), page1_html, css, "timeline about")
    rows = f"""
    <table style="width:100%">
    <tr><td style="width:92pt;text-align:right;color:#6b6b6b">2021 - Present</td>
    <td><b>Full-Stack Developer</b> - Pixelforge Studio, Medellin<br>
    {bullets([
        "Lead developer on a React and TypeScript design-system used by nine client projects; component adoption cut UI delivery time by about a third.",
        "Built Node.js and GraphQL back-ends on PostgreSQL for two e-commerce launches (peak 12,000 concurrent shoppers).",
        "Set up Lighthouse budgets in CI; mean Largest Contentful Paint improved from 3.9 s to 1.8 s.",
        "Run the weekly accessibility clinic; all new components meet WCAG 2.1 AA.",
    ])}</td></tr>
    <tr><td style="width:92pt;text-align:right;color:#6b6b6b">2019 - 2021</td>
    <td><b>Developer</b> - Andes Fintech, Bogota<br>
    {bullets([
        "Delivered the web onboarding flow for a digital wallet (KYC, document upload, status tracking); conversion rose from 41% to 63%.",
        "Wrote integration tests with Cypress covering 85% of critical user journeys.",
        "Migrated the front-end from AngularJS to React incrementally, over eight months, without a freeze.",
    ])}</td></tr>
    </table>
    """
    box(page1, (40, 196, 560, 812), rows, css, "timeline rows 1")

    page2 = doc.new_page(width=A4.width, height=A4.height)
    page2.draw_line(pymupdf.Point(138, 40), pymupdf.Point(138, 215), color=(0.7, 0.55, 0.78), width=1.4)
    rows2 = f"""
    <table style="width:100%">
    <tr><td style="width:92pt;text-align:right;color:#6b6b6b">2018 - 2019</td>
    <td><b>Web Developer (freelance)</b> - Self-employed, Bogota<br>
    {bullets([
        "Built and maintained WordPress and Laravel sites for 11 small businesses.",
        "Introduced a reusable checkout plugin that two clients kept using after the contract ended.",
    ])}</td></tr>
    <tr><td style="width:92pt;text-align:right;color:#6b6b6b">2013 - 2018</td>
    <td><b>Ingenieria de Sistemas</b> (Bachelor of Systems Engineering) - Universidad Nacional de Colombia, Bogota<br>
    Graduated with honours; capstone: offline-first attendance app for rural schools.</td></tr>
    <tr><td style="width:92pt;text-align:right;color:#6b6b6b">2022</td>
    <td><b>AWS Certified Developer - Associate</b></td></tr>
    <tr><td style="width:92pt;text-align:right;color:#6b6b6b">2023</td>
    <td><b>Meta Front-End Developer Professional Certificate</b></td></tr>
    </table>
    <h2>SKILLS</h2>
    <p><b>Front-end:</b> React, TypeScript, Next.js, HTML, CSS, Storybook<br>
    <b>Back-end:</b> Node.js, Express, GraphQL, PostgreSQL, Redis<br>
    <b>Tooling:</b> Git, Docker, GitHub Actions, Cypress, Jest<br>
    <b>Cloud:</b> AWS (Lambda, S3, CloudFront)</p>
    <h2>LANGUAGES</h2>
    <p>Spanish (native), English (professional working proficiency), Portuguese (basic)</p>
    """
    box(page2, (40, 40, 560, 800), rows2, css, "timeline rows 2")
    data = doc.tobytes()
    doc.close()
    return finish(data, "Camila Rojas - CV", "Camila Rojas")


# --- 6. Tobias Wagner: dense three-page CV with a running header -----------------------


def resume_dense() -> bytes:
    css = BASE_CSS.replace("10.5pt", "9.3pt") + "h1 { font-size: 19pt; } h2 { font-size: 10pt; }"
    html = f"""
    <h1>TOBIAS WAGNER</h1>
    <p>Lead Software Architect | Munich, Germany | tobias.wagner@example.com | +49 151 5550 1234 | xing.example.com/tobias-wagner</p>
    <h2>KEY ACHIEVEMENTS</h2>
    <ul>
    <li>Architected the policy-administration platform for a top-5 German insurer, now serving 3.2 million contracts and 41 internal teams.</li>
    <li>Reduced yearly infrastructure and licence cost by EUR 1.9M by retiring a mainframe batch system in favour of event-driven Java services.</li>
    <li>Grew a distributed architecture guild from 6 to 55 engineers and introduced architecture decision records company-wide.</li>
    <li>Delivered three regulatory programmes (Solvency II reporting, GDPR erasure, DORA resilience testing) on schedule.</li>
    </ul>
    <h2>PROFESSIONAL EXPERIENCE</h2>
    {jobs_html([
        {"company": "Bavaria Insure AG", "place": "Munich", "dates": "April 2019 - Present", "roles": [
            {"title": "Lead Software Architect", "dates": "January 2022 - Present", "bullets": [
                "Set the target architecture for 14 bounded contexts using domain-driven design, Kafka and Spring Boot 3; coordinate eight squads.",
                "Introduced contract testing (Pact) across 60 service boundaries; integration defects in production fell by 57%.",
                "Led a performance programme on the quote engine: p95 latency from 2.8 s to 340 ms through caching, query tuning and JVM profiling.",
                "Chair the architecture review board and own the technology radar.",
                "Sponsor of the zero-downtime release initiative (feature flags, canary releases, database expand/contract migrations).",
            ]},
            {"title": "Senior Software Architect", "dates": "April 2019 - December 2021", "bullets": [
                "Designed the event-sourced claims-handling core and its migration path from the legacy Oracle schema.",
                "Defined security architecture (OAuth 2.0, mTLS, field-level encryption) approved by the data-protection office.",
                "Built the reference implementation reused by 12 teams, with Gradle plugins and golden-path templates.",
            ]},
        ]},
        {"company": "Rhein Logistics GmbH", "place": "Cologne", "dates": "January 2015 - March 2019", "roles": [
            {"title": "Senior Java Developer", "bullets": [
                "Built the route-optimisation service (Java 11, Spring, PostgreSQL, OR-Tools) that saved 6% of fleet mileage across 2,400 vehicles.",
                "Led the move from WebSphere to containers on OpenShift for 22 applications.",
                "Implemented the tracking API used by 300 customers; sustained 5,000 requests per second.",
                "Coached a team of six through the adoption of trunk-based development and continuous delivery.",
            ]}]},
        {"company": "Alpenbank Systems", "place": "Munich", "dates": "September 2012 - December 2014", "roles": [
            {"title": "Java Developer", "bullets": [
                "Developed payment-order processing modules in Java EE with Oracle and MQ.",
                "Wrote the automated acceptance suite (Selenium, Cucumber) that gated every release.",
                "Took part in the Basel III reporting project as the developer responsible for the risk-aggregation job.",
            ]}]},
    ], date_style="table")}
    <h2>SELECTED PROJECTS</h2>
    <table style="width:100%">
    <tr><td><b>Claims 2.0</b></td><td>Event-sourced claims platform; Kafka, Spring Boot, PostgreSQL; 18 months; team of 40</td></tr>
    <tr><td><b>Quote Engine Rewrite</b></td><td>JVM performance, caching strategy and rating-table service; 9 months; team of 12</td></tr>
    <tr><td><b>Fleet Optimiser</b></td><td>Constraint-based routing with OR-Tools; 14 months; team of 8</td></tr>
    <tr><td><b>Resilience Testing (DORA)</b></td><td>Chaos experiments, game days and evidence packs for regulators; 6 months</td></tr>
    </table>
    <h2>TECHNICAL SKILLS</h2>
    <p><b>Languages:</b> Java (8 to 21), Kotlin, SQL, Python, TypeScript<br>
    <b>Frameworks:</b> Spring Boot, Spring Cloud, Hibernate, Micronaut<br>
    <b>Architecture:</b> Domain-driven design, event sourcing, CQRS, microservices, hexagonal architecture<br>
    <b>Messaging and data:</b> Kafka, RabbitMQ, PostgreSQL, Oracle, MongoDB, Elasticsearch<br>
    <b>Platforms and delivery:</b> Kubernetes, OpenShift, AWS, Docker, Gradle, Jenkins, GitLab CI, Terraform<br>
    <b>Quality:</b> JUnit, Pact, Testcontainers, Gatling, SonarQube</p>
    <h2>EDUCATION</h2>
    <table style="width:100%"><tr><td><b>B.Sc. Computer Science</b>, Technical University of Munich</td><td style="text-align:right;width:120pt">2008 - 2012</td></tr></table>
    <h2>CERTIFICATIONS AND TRAINING</h2>
    <ul>
    <li>TOGAF 9 Certified, 2021</li>
    <li>Oracle Certified Professional, Java SE 11 Developer, 2020</li>
    <li>AWS Certified Solutions Architect - Associate, 2019</li>
    <li>Professional Scrum Master II (PSM II), 2020</li>
    <li>Domain-Driven Design Europe workshop series, 2018 - 2023</li>
    </ul>
    <h2>LANGUAGES</h2>
    <p>German (native), English (fluent), French (conversational)</p>
    <h2>PUBLICATIONS AND TALKS</h2>
    <ul>
    <li>"Strangling the mainframe with events", JAX Conference, Mainz, 2023</li>
    <li>"Architecture decision records in practice", Software Architecture Summit, Berlin, 2022</li>
    <li>Co-author, "Resilience for regulated systems", Bavaria Insure Engineering Blog, 2024</li>
    </ul>
    """
    data = flow(html, css, [[margins(left=44, top=50, right=44, bottom=56)]])
    return finish(data, "Tobias Wagner - CV", "Tobias Wagner",
                  footer("T. Wagner - CV", running_header="Tobias Wagner - Lead Software Architect",
                         from_page=1))


# --- 7. Fatima Al-Mansouri: text flowing across two equal columns ----------------------


def resume_columns() -> bytes:
    css = BASE_CSS.replace("10.5pt", "9.8pt") + "h1 { font-size: 22pt; text-align: center; } " \
        "h2 { color: #7a1f2b; }"
    left = pymupdf.Rect(40, 120, 290, 800)
    right = pymupdf.Rect(312, 120, 555, 800)
    header_html = ""  # the header is drawn separately so it spans both columns
    body = f"""
    <h2>PROFILE</h2>
    <p>Security engineer with eight years of experience protecting banking and telecom infrastructure. Combines hands-on penetration testing with detection engineering and security programme management. Known for turning audit findings into automated controls.</p>
    <h2>EXPERIENCE</h2>
    {jobs_html([
        {"company": "Gulf Secure Networks", "place": "Dubai", "dates": "Sep 2018 - Present", "roles": [
            {"title": "Senior Security Engineer", "bullets": [
                "Lead detection engineering for a 24x7 SOC; wrote 220 Sigma and Splunk detections mapped to MITRE ATT&CK, lifting coverage of relevant techniques from 38% to 74%.",
                "Run quarterly red-team exercises; remediation time for critical findings fell from 41 to 9 days.",
                "Built an infrastructure-as-code security scanner (Python, Terraform, OPA) that blocks risky changes in 90+ pipelines.",
                "Handled 3 major incidents as incident commander, including a business-email-compromise attempt contained within 50 minutes.",
            ]}]},
        {"company": "Emirates Digital Bank", "place": "Dubai", "dates": "Mar 2015 - Aug 2018", "roles": [
            {"title": "Security Analyst", "bullets": [
                "Monitored and triaged alerts in QRadar; tuned rules to cut false positives by 45%.",
                "Performed web-application and mobile penetration tests on 30 customer-facing applications.",
                "Prepared evidence for PCI DSS and ISO 27001 audits with zero major non-conformities.",
            ]}]},
    ], date_style="plain")}
    <h2>EDUCATION</h2>
    <p><b>B.Sc. in Information Security</b><br>Zayed University, Abu Dhabi<br>2010 - 2014</p>
    <h2>CERTIFICATIONS</h2>
    <ul><li>CISSP, 2019</li><li>Offensive Security Certified Professional (OSCP), 2021</li>
    <li>Certified Ethical Hacker (CEH), 2016</li><li>AWS Certified Security - Specialty, 2022</li>
    <li>GIAC Certified Incident Handler (GCIH), 2020</li></ul>
    <h2>SKILLS</h2>
    <p><b>Offensive:</b> Burp Suite, Metasploit, Nmap, Cobalt Strike<br>
    <b>Defensive:</b> Splunk, QRadar, CrowdStrike, Wazuh, Sigma, YARA<br>
    <b>Cloud:</b> AWS, Azure, Terraform, Open Policy Agent<br>
    <b>Programming:</b> Python, Bash, PowerShell, SQL<br>
    <b>Frameworks:</b> MITRE ATT&CK, NIST CSF, ISO 27001, PCI DSS</p>
    <h2>LANGUAGES</h2>
    <p>Arabic (native), English (fluent), Hindi (conversational)</p>
    <h2>VOLUNTEERING</h2>
    <p>Mentor, Women in Cyber Middle East (2020 - Present). Capture-the-flag organiser, Zayed University (2019 - 2022).</p>
    """
    story_html = header_html + body
    data = flow(story_html, css, [[left, right], [left, right]])
    pre = pymupdf.open(stream=data, filetype="pdf")
    p0 = pre[0]
    p0.draw_rect(pymupdf.Rect(0, 0, A4.width, 106), color=None, fill=(0.48, 0.12, 0.17), overlay=False)
    box(p0, (40, 10, 555, 102), """<p style="font-size:25pt;color:#ffffff;text-align:center"><b>Fatima Al-Mansouri</b></p>
        <p style="font-size:8.6pt;color:#f4dde0;text-align:center">Senior Security Engineer | Dubai, UAE | fatima.almansouri@example.com | +971 50 555 0166</p>""",
        "body { font-family: sans-serif; } p { margin: 3pt 0; }", "columns header")
    data = pre.tobytes()
    pre.close()
    return finish(data, "Fatima Al-Mansouri - Resume", "Fatima Al-Mansouri")


# --- 8. Arjun Mehta: project cards beside the experience column ------------------------


def resume_cards() -> bytes:
    doc = pymupdf.open()
    css = BASE_CSS.replace("10.5pt", "9.8pt") + "h2 { color: #8a4b00; }"
    card_css = "body { font-family: sans-serif; font-size: 9.4pt; color: #222222; } " \
               "p { margin: 2pt 0; } h3 { font-size: 9.5pt; margin: 0 0 3pt 0; color: #8a4b00; }"
    p1 = doc.new_page(width=A4.width, height=A4.height)
    p1.draw_rect(pymupdf.Rect(0, 0, A4.width, 8), color=None, fill=(0.9, 0.55, 0.1))
    box(p1, (40, 16, 400, 114), """<p style="font-size:24pt"><b>Arjun Mehta</b></p>
        <p style="font-size:12pt;color:#8a4b00">Senior Mobile Engineer (iOS and Android)</p>""",
        "body { font-family: sans-serif; } p { margin: 3pt 0; }", "cards name")
    box(p1, (400, 22, 575, 112), """<p style="font-size:9.6pt">Toronto, Canada</p>
        <p style="font-size:9.6pt">arjun.mehta@example.com</p><p style="font-size:9.6pt">+1 416 555 0123</p>
        <p style="font-size:9.6pt">github.com/arjunm-example</p>""",
        "body { font-family: sans-serif; }", "cards contact")
    main = f"""
    <h2>SUMMARY</h2>
    <p>Mobile engineer with seven years of experience building consumer apps with millions of downloads. Deep in Kotlin, Swift and cross-platform architecture; passionate about startup time, offline behaviour and release quality.</p>
    <h2>WORK HISTORY</h2>
    {jobs_html([
        {"company": "Maple Rides Inc.", "place": "Toronto", "dates": "05/2022 - Present", "roles": [
            {"title": "Senior Mobile Engineer", "bullets": [
                "Own the rider app (Kotlin and Swift, 4.7 stars, 3.1M monthly users); cold-start time cut from 3.4 s to 1.6 s.",
                "Led the move to a shared Kotlin Multiplatform domain layer, removing about 30% of duplicated business logic.",
                "Built the offline trip-request queue; failed requests in poor coverage areas dropped by 58%.",
                "Reduced crash-free-session gap with staged rollouts and Firebase Crashlytics triage; crash rate fell from 1.1% to 0.3%.",
            ]}]},
        {"company": "Northern Fitness Labs", "place": "Toronto", "dates": "01/2019 - 04/2022", "roles": [
            {"title": "Mobile Engineer (Android lead)", "bullets": [
                "Rebuilt the Android app in Kotlin with Jetpack Compose; 1.2M installs, rating 3.9 to 4.6.",
                "Integrated Bluetooth heart-rate straps and Google Fit; sensor pairing success reached 97%.",
                "Set up CI/CD with Bitrise and Fastlane; release cadence moved from monthly to weekly.",
            ]}]},
        {"company": "Pixel Pioneer (start-up)", "place": "Waterloo", "dates": "06/2017 - 12/2018", "roles": [
            {"title": "Android Developer", "bullets": [
                "Shipped three puzzle games to Google Play (combined 2M downloads).",
                "Implemented analytics and A/B testing that improved day-7 retention from 14% to 21%.",
            ]}]},
    ])}
    <h2>EDUCATION</h2>
    <p><b>B.A.Sc. Software Engineering</b>, University of Waterloo (co-op), 2013 - 2017</p>
    """
    box(p1, (40, 124, 355, 812), main, css, "cards main")
    cards = [
        ("RideKit SDK", "Open-source Kotlin Multiplatform library for trip state machines. 900 stars, used in two production apps."),
        ("Offline Queue", "Persistent request queue with retry policies and conflict handling for flaky networks (iOS and Android)."),
        ("PulseBand", "Companion app for Bluetooth heart-rate straps; real-time chart rendering at 60 fps on mid-range phones."),
    ]
    box(p1, (372, 122, 560, 142), "<h3>PROJECTS</h3>", card_css)
    y = 146
    for title, text in cards:
        p1.draw_rect(pymupdf.Rect(372, y, 560, y + 112), color=(0.82, 0.74, 0.62), fill=(0.99, 0.96, 0.91), width=1)
        box(p1, (380, y + 6, 552, y + 108), f"<h3>{esc(title)}</h3><p>{esc(text)}</p>", card_css, title)
        y += 124
    box(p1, (372, y + 6, 560, y + 130), """<h3>SKILLS</h3><p><b>Android:</b> Kotlin, Jetpack Compose, Coroutines, Room</p>
        <p><b>iOS:</b> Swift, SwiftUI, Combine</p><p><b>Shared:</b> Kotlin Multiplatform, GraphQL, REST</p>
        <p><b>Tools:</b> Fastlane, Bitrise, Firebase, Sentry</p>""", card_css, "cards skills")

    p2 = doc.new_page(width=A4.width, height=A4.height)
    box(p2, (40, 40, 560, 300), """
        <h2>CERTIFICATIONS</h2>
        <ul><li>Associate Android Developer, Google, 2019</li>
        <li>Apple Developer Academy Swift Programme, 2021</li></ul>
        <h2>TALKS AND WRITING</h2>
        <ul><li>"Cold start budgets", Android Toronto Meetup, 2023</li>
        <li>"Kotlin Multiplatform without the hype", personal blog, 2022</li></ul>
        <h2>INTERESTS</h2><p>Cycling, board games, volunteering as a coding mentor for newcomers.</p>""",
        css, "cards p2")
    data = doc.tobytes()
    doc.close()
    return finish(data, "Arjun Mehta - Resume", "Arjun Mehta")


# --- 9. Isabella Rossi: scanned two-column resume (image only, needs OCR) --------------


def resume_scanned() -> bytes:
    return resume_sidebar_variant()


def resume_sidebar_variant() -> bytes:
    """A different person in the sidebar layout, then rasterised like a photocopy."""
    doc = pymupdf.open()
    side_css = "body { font-family: sans-serif; font-size: 10pt; color: #222222; } " \
               "h3 { font-size: 10pt; margin: 0 0 4pt 0; color: #123; } p { margin: 2pt 0; } " \
               "ul { margin: 0; } li { margin-bottom: 2pt; }"
    main_css = BASE_CSS.replace("10.5pt", "10.4pt")
    page = doc.new_page(width=A4.width, height=A4.height)
    page.draw_rect(pymupdf.Rect(0, 0, 190, A4.height), color=None, fill=(0.88, 0.9, 0.93))
    box(page, (16, 40, 178, 800), """
        <h3>CONTACT</h3><p>Milan, Italy</p><p>isabella.rossi@example.com</p><p>+39 333 555 0142</p>
        <p>linkedin.com/in/isabella-rossi-example</p><br>
        <h3>SKILLS</h3>
        <ul><li>Test automation</li><li>Selenium and Playwright</li><li>API testing (Postman, REST Assured)</li>
        <li>Java and TypeScript</li><li>CI/CD (GitLab CI, Jenkins)</li><li>Performance testing (k6, JMeter)</li>
        <li>Test strategy and coaching</li></ul><br>
        <h3>CERTIFICATIONS</h3>
        <ul><li>ISTQB Advanced Test Automation Engineer, 2022</li><li>ISTQB Foundation Level, 2018</li></ul><br>
        <h3>LANGUAGES</h3><p>Italian (native)<br>English (fluent)<br>German (basic)</p>""",
        side_css, "scan side")
    main = f"""
    <h1>Isabella Rossi</h1>
    <p style="font-size:12pt;color:#555555">QA Automation Lead</p>
    <h2>PROFILE</h2>
    <p>Quality engineer with six years of experience building test automation that teams actually trust. Leads a group of five testers and developers, owns the quality strategy for a payments web app, and cares about fast, deterministic pipelines.</p>
    <h2>EXPERIENCE</h2>
    {jobs_html([
        {"company": "Lombardia Software", "place": "Milan", "dates": "January 2020 - Present", "roles": [
            {"title": "QA Automation Lead", "bullets": [
                "Designed a Playwright and TypeScript end-to-end framework running 900 tests in 11 minutes across 12 parallel workers.",
                "Reduced flaky tests from 14% to under 1% with retry analytics, test-data isolation and contract tests.",
                "Introduced k6 performance gates in the pipeline; two regressions were blocked before release.",
                "Lead a team of five and run the monthly quality review with product management.",
            ]}]},
        {"company": "Navigli Digital", "place": "Milan", "dates": "September 2017 - December 2019", "roles": [
            {"title": "QA Engineer", "bullets": [
                "Automated 70% of the regression suite with Selenium and Java, cutting release testing from 5 days to 1.",
                "Wrote REST Assured API tests for 140 endpoints and integrated them into Jenkins.",
                "Triaged production defects with support, shortening average fix time by 30%.",
            ]}]},
    ], date_style="plain")}
    <h2>EDUCATION</h2>
    <p><b>B.Sc. in Computer Science</b>, University of Milan, 2014 - 2017</p>
    """
    box(page, (206, 40, 560, 800), main, main_css, "scan main")
    data = doc.tobytes()
    doc.close()
    return rasterise(finish(data, "scan", "Isabella Rossi"), dpi=170)


RESUMES = [
    ("resume_01_ananya_krishnan_platform_engineer.pdf", resume_classic),
    ("resume_02_daniel_okafor_ml_engineer.pdf", resume_sidebar),
    ("resume_03_mei_tanaka_data_scientist.pdf", resume_academic),
    ("resume_04_lars_johansson_sre.pdf", resume_band),
    ("resume_05_camila_rojas_fullstack.pdf", resume_timeline),
    ("resume_06_tobias_wagner_architect.pdf", resume_dense),
    ("resume_07_fatima_almansouri_security.pdf", resume_columns),
    ("resume_08_arjun_mehta_mobile.pdf", resume_cards),
    ("resume_09_isabella_rossi_qa_scanned.pdf", resume_scanned),
]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=True)
    for filename, build in RESUMES:
        data = build()
        (out / filename).write_bytes(data)
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            text_chars = sum(len(p.get_text()) for p in doc)
            print(f"{filename:48s} {doc.page_count} page(s), {text_chars:6d} text chars, "
                  f"{len(data) / 1024:6.0f} KB")
    for line in warnings:
        print("WARNING:", line)


if __name__ == "__main__":
    main()
