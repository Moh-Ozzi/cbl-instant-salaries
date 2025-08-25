# CBL Instant Salaries Dashboard

An interactive web dashboard for the **Central Bank of Libya’s (CBL) Instant Salaries service**.  
CBL currently shares this data as an Excel file on their public website. This project transforms that file into a live, interactive dashboard using [Dash](https://dash.plotly.com/).

---

## Overview
- Displays key summary indicators:
  - Total citizens
  - Total bank accounts
  - Accounts not entered
  - Average entry percentage (weighted)
- Interactive horizontal bar charts by **region** and by **organization**.
- Filter controls for region and organization, plus sorting options.

---

## Tech Stack
- [Dash](https://dash.plotly.com/) (Plotly + Flask)  
- [dash-bootstrap-components](https://dash-bootstrap-components.opensource.faculty.ai/)  
- [dash-bootstrap-templates](https://github.com/AnnMarieW/dash-bootstrap-templates)  
- [pandas](https://pandas.pydata.org/) for data processing  
- Hosted on [Render](https://render.com/)  

---

## Deployment
The app is deployed on Render and automatically updates on every push to GitHub.  

### Run locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
python app.py