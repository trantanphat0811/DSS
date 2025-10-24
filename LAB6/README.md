# DSS Lab6 – Flask Plotly Dashboard

Interactive local dashboard built with Flask + Plotly to visualize candidate data (100 randomized records per run). The app renders multiple charts and an insight panel beside each chart.

## Features
- Data ingest from CSV/XLSX (fallback sample if missing)
- Auto-normalize to exactly 100 candidates (random sample/upsample)
- 6+ visuals with a modern, consistent style:
  - Pie: Candidates by position (donut)
  - Bar: Candidates by education level
  - Combo: Candidates and suitable rate by years of experience
  - Area: Cumulative candidates by experience
  - Heatmap: Position × binned experience (count)
  - Horizontal bar: Average years of experience by position
  - Table: First 100 records (Bootstrap table)
- Insight cards: auto-generated bullet points per chart
- Bootstrap layout, responsive 8/4 split (chart/report)

## Tech Stack
- Python 3.9+
- Flask
- Pandas
- Plotly (graph_objs + pio)
- Bootstrap 5 (CDN)

## Project Structure
```
LAB6/
├─ bt1.py                 # Flask app and chart generation
├─ templates/
│  └─ index.html          # Bootstrap layout + insight panels
├─ data/                  # Place your dataset(s) here
│  ├─ candidates.csv      # Optional input
│  └─ candidates.xlsx     # Optional input
├─ statics/               # Optional styles if needed
└─ README.md
```

## Data Requirements
Expected columns (missing ones will be created with defaults):
- candidate_id, years_experience, education_level, skills, experience_description, position_applied, suitable

Notes:
- If more than 100 rows: app randomly samples 100.
- If fewer than 100: app upsamples with replacement and generates unique ids.

## Setup & Run
1) (Optional) Create and activate a virtual environment.
2) Install dependencies:
```bash
pip install flask pandas plotly
```
3) Run the server (choose any free port):
```bash
PORT=5000 python3 bt1.py
```
4) Open the app:
```
http://127.0.0.1:5000/
```

## Common Issues
- Port in use: run on another port, e.g. `PORT=5057 python3 bt1.py`.
- Blank charts: ensure your dataset has the expected columns; the app will still render with fallbacks.

## Customization
- Colors/labels: tweak in make_plot_divs(...) inside bt1.py.
- Insight wording: adjust make_insights(...) in bt1.py.
- Layout: edit Bootstrap grid in templates/index.html.

## License
For educational use in DSS Lab6.

