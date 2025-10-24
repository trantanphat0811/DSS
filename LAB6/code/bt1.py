from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import os

app = Flask(__name__)

DATA_CSV = "candidates.csv"
DATA_XLSX = "candidates.xlsx"

def load_and_prepare():
    # Try CSV then XLSX, otherwise create example dataset
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
    elif os.path.exists(DATA_XLSX):
        df = pd.read_excel(DATA_XLSX)
    else:
        # Fallback example small dataset (you can remove this if you always have a file)
        data = [
            {"candidate_id":"CAND_001","years_experience":6,"education_level":"master","skills":"python,sql","position_applied":"data_analyst","suitable":1},
            {"candidate_id":"CAND_002","years_experience":11,"education_level":"associate","skills":"kubernetes,react","position_applied":"developer","suitable":1},
            {"candidate_id":"CAND_003","years_experience":8,"education_level":"associate","skills":"azure,ml","position_applied":"data_scientist","suitable":0},
            {"candidate_id":"CAND_004","years_experience":17,"education_level":"associate","skills":"javascript,ux","position_applied":"designer","suitable":1},
            {"candidate_id":"CAND_005","years_experience":14,"education_level":"phd","skills":"git,leadership","position_applied":"manager","suitable":1},
            {"candidate_id":"CAND_006","years_experience":15,"education_level":"phd","skills":"docker,sql","position_applied":"devops_engineer","suitable":0},
            {"candidate_id":"CAND_007","years_experience":6,"education_level":"high_school","skills":"communication,scrum","position_applied":"designer","suitable":1},
            {"candidate_id":"CAND_008","years_experience":11,"education_level":"master","skills":"leadership,aws","position_applied":"designer","suitable":1},
            {"candidate_id":"CAND_009","years_experience":4,"education_level":"bachelor","skills":"sql,excel","position_applied":"analyst","suitable":0},
            {"candidate_id":"CAND_010","years_experience":3,"education_level":"bachelor","skills":"python,react","position_applied":"frontend_developer","suitable":1},
        ]
        df = pd.DataFrame(data)

    # Standardize column names (strip & lower underscores)
    df.columns = [c.strip() for c in df.columns]

    # Ensure expected columns exist
    required = ["candidate_id","years_experience","education_level","skills","experience_description","position_applied","suitable"]
    for c in required:
        if c not in df.columns:
            # add missing with defaults
            if c == "candidate_id":
                df["candidate_id"] = [f"auto_{i}" for i in range(len(df))]
            elif c == "years_experience":
                df[c] = pd.to_numeric(df.get(c, 0)).fillna(0).astype(int)
            elif c == "suitable":
                df[c] = df.get(c, 0).fillna(0).astype(int)
            else:
                df[c] = df.get(c, "")

    # Clean numeric columns
    df["years_experience"] = pd.to_numeric(df["years_experience"], errors="coerce").fillna(0).astype(int)
    df["suitable"] = pd.to_numeric(df["suitable"], errors="coerce").fillna(0).astype(int)

    # Ensure exactly 100 candidates (randomized order/selection)
    target_n = 100
    current_n = len(df)
    if current_n < target_n:
        # Upsample with replacement (random)
        extra = df.sample(n=target_n - current_n, replace=True).copy().reset_index(drop=True)
        # Assign new unique candidate_id for duplicated rows
        start_idx = current_n + 1
        extra["candidate_id"] = [f"CAND_{i:04d}" for i in range(start_idx, start_idx + len(extra))]
        df = pd.concat([df.reset_index(drop=True), extra], ignore_index=True)
        # Shuffle the final 100 randomly
        df = df.sample(frac=1.0).reset_index(drop=True)
    elif current_n > target_n:
        # Downsample to exactly 100 (random)
        df = df.sample(n=target_n).reset_index(drop=True)

    # candidate_count per position
    pos_counts = df.groupby("position_applied")["candidate_id"].count().reset_index(name="candidate_count")
    df = df.merge(pos_counts, on="position_applied", how="left")

    # suitable_rate per position: fraction of suitable candidates in that position
    suitable_pos = df.groupby("position_applied")["suitable"].mean().reset_index(name="suitable_rate")
    df = df.merge(suitable_pos, on="position_applied", how="left")

    # Also create aggregates for plots
    agg_position = df.groupby("position_applied").agg(
        candidate_count=("candidate_id", "count"),
        suitable_rate=("suitable", "mean"),
    ).sort_values("candidate_count", ascending=False).reset_index()

    agg_education = df.groupby("education_level").agg(candidate_count=("candidate_id","count")).reset_index()
    agg_experience = df.groupby("years_experience").agg(candidate_count=("candidate_id","count"),
                                                       suitable_rate=("suitable","mean")).reset_index().sort_values("years_experience")

    # pivot for stacked bar: position vs education (counts)
    pivot_pos_ed = df.pivot_table(index="position_applied", columns="education_level", values="candidate_id", aggfunc="count", fill_value=0)

    return df, agg_position, agg_education, agg_experience, pivot_pos_ed

def make_plot_divs(df, agg_position, agg_education, agg_experience, pivot_pos_ed):
    plots = {}

    # 1) Pie: positions (top positions)
    pie = go.Pie(
        labels=agg_position["position_applied"],
        values=agg_position["candidate_count"],
        hole=0.4,
        textinfo="percent+label",
        insidetextorientation="radial",
        marker=dict(colors=["#c7d2fe","#a5b4fc","#93c5fd","#86efac","#fde68a","#fca5a5","#fcd34d","#99f6e4"]) 
    )
    fig1 = go.Figure(pie)
    fig1.update_layout(
        title=dict(text="Candidates by Position", x=0.02, xanchor="left", font=dict(size=24, color="#111827")),
        template="plotly_white",
        legend_title_text="Position",
        margin=dict(t=60, r=20, b=60, l=20)
    )
    fig1.add_annotation(text="Share of candidates by applied position", x=0.02, y=-0.18, xref="paper", yref="paper", showarrow=False, font=dict(color="#6b7280"))
    plots["pie_positions"] = pio.to_html(fig1, full_html=False, include_plotlyjs="cdn")

    # 2) Bar: education levels
    bar = go.Bar(
        x=agg_education["education_level"],
        y=agg_education["candidate_count"],
        text=agg_education["candidate_count"],
        textposition="outside",
        marker_color="#93c5fd",
        marker_line_width=0
    )
    fig2 = go.Figure(bar)
    fig2.update_layout(
        title=dict(text="Candidates by Education Level", x=0.02, xanchor="left", font=dict(size=24, color="#111827")),
        xaxis=dict(title="Education Level", gridcolor="#e5e7eb"),
        yaxis=dict(title="Count", gridcolor="#e5e7eb"),
        template="plotly_white",
        uniformtext_minsize=10,
        uniformtext_mode="hide",
        margin=dict(t=60, r=20, b=60, l=60)
    )
    fig2.add_annotation(text="Counts per education level", x=0.02, y=-0.18, showarrow=False, xref="paper", yref="paper", font=dict(color="#6b7280"))
    plots["bar_education"] = pio.to_html(fig2, full_html=False, include_plotlyjs=False)

    # 3) Line + Bars combo: years_experience vs candidate_count and suitable_rate (secondary axis)
    fig3 = go.Figure()
    # Bars with value labels on top, soft color
    fig3.add_trace(go.Bar(
        x=agg_experience["years_experience"],
        y=agg_experience["candidate_count"],
        name="candidate_count",
        marker_color="#c7d2fe",  # light periwinkle
        marker_line_width=0,
        text=agg_experience["candidate_count"],
        textposition="outside",
        textfont=dict(color="#111827", size=12)
    ))
    # Smooth line with markers on secondary axis
    fig3.add_trace(go.Scatter(
        x=agg_experience["years_experience"],
        y=agg_experience["suitable_rate"],
        name="suitable_rate",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#84cc16", width=3, shape="spline", smoothing=0.8),
        marker=dict(size=7, color="#a3e635", line=dict(width=0)),
        hovertemplate="Years=%{x}<br>Rate=%{y:.2f}<extra></extra>"
    ))
    fig3.update_layout(
        title=dict(text="Candidate Count and Suitability Rate by Years of Experience", x=0.02, xanchor="left", font=dict(size=28, family="Inter, Arial, sans-serif", color="#111827")),
        xaxis=dict(title="Years of Experience", gridcolor="#e5e7eb", zeroline=False, tickmode="linear"),
        yaxis=dict(title="Candidate Count", gridcolor="#e5e7eb", zeroline=False),
        yaxis2=dict(title="", overlaying="y", side="right", range=[0,1], tickformat=".1f", showgrid=False),
        bargap=0.2,
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(t=80, r=20, b=80, l=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02)
    )
    fig3.add_annotation(text="Bar: candidate_count · Line: suitable_rate (0–1)", x=0.02, xref="paper", y=-0.2, yref="paper", showarrow=False, font=dict(color="#6b7280"))
    plots["combo_experience"] = pio.to_html(fig3, full_html=False, include_plotlyjs=False)

    # 4) Area: cumulative distribution of experience (area)
    agg_experience["cum_count"] = agg_experience["candidate_count"].cumsum()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=agg_experience["years_experience"], y=agg_experience["cum_count"], fill='tozeroy', name="Cumulative candidates", line=dict(color="#6366f1", width=3)))
    fig4.update_layout(
        title=dict(text="Cumulative Candidates by Experience", x=0.02, xanchor="left", font=dict(size=24, color="#111827")),
        xaxis=dict(title="Years of Experience", gridcolor="#e5e7eb"),
        yaxis=dict(title="Cumulative Count", gridcolor="#e5e7eb"),
        template="plotly_white",
        margin=dict(t=60, r=20, b=60, l=60)
    )
    # Annotate max point
    if not agg_experience.empty:
        max_x = agg_experience["years_experience"].iloc[-1]
        max_y = agg_experience["cum_count"].iloc[-1]
        fig4.add_annotation(x=max_x, y=max_y, text=f"Max: {int(max_y)}", showarrow=True, arrowhead=2)
    fig4.add_annotation(text="Cumulative sum across experience", x=0.5, y=-0.18, showarrow=False, xref="paper", yref="paper")
    plots["area_cumulative"] = pio.to_html(fig4, full_html=False, include_plotlyjs=False)

    # 5) Heatmap: Position vs Binned Years of Experience (suitable for this dataset)
    bins = [0, 2, 5, 8, 11, 20]
    labels = ["0–2", "3–5", "6–8", "9–11", "12+" ]
    df["years_bin"] = pd.cut(df["years_experience"], bins=bins, labels=labels, right=True, include_lowest=True)
    hm = df.pivot_table(index="position_applied", columns="years_bin", values="candidate_id", aggfunc="count", fill_value=0)

    fig5 = go.Figure(data=go.Heatmap(
        z=hm.values,
        x=list(hm.columns.astype(str)),
        y=list(hm.index.astype(str)),
        colorscale="Blues",
        colorbar=dict(title="Count")
    ))
    fig5.update_layout(
        title=dict(text="Position vs Years of Experience (Count Heatmap)", x=0.02, xanchor="left", font=dict(size=24, color="#111827")),
        xaxis=dict(title="Years of Experience (bins)", gridcolor="#e5e7eb"),
        yaxis=dict(title="Position", gridcolor="#e5e7eb"),
        template="plotly_white",
        margin=dict(t=60, r=20, b=60, l=100)
    )
    # Optional annotations of counts on heatmap
    annotations = []
    for i, y_val in enumerate(hm.index):
        for j, x_val in enumerate(hm.columns):
            annotations.append(dict(
                x=str(x_val), y=str(y_val), text=str(hm.values[i][j]),
                xref='x', yref='y', showarrow=False, font=dict(color='#111827', size=10)
            ))
    fig5.update_layout(annotations=annotations)
    fig5.add_annotation(text="Count of candidates per position and experience bin", x=0.02, y=-0.18, showarrow=False, xref="paper", yref="paper", font=dict(color="#6b7280"))
    plots["scatter_individuals"] = pio.to_html(fig5, full_html=False, include_plotlyjs=False)

    # 6) Alternative: Average Years Experience by Position (horizontal bar)
    avg_pos = df.groupby("position_applied").agg(
        avg_years=("years_experience", "mean"),
        candidate_count=("candidate_id", "count")
    ).reset_index().sort_values("avg_years", ascending=False)

    fig6 = go.Figure(go.Bar(
        y=avg_pos["position_applied"],
        x=avg_pos["avg_years"].round(2),
        orientation='h',
        text=avg_pos["avg_years"].round(2),
        textposition='outside',
        marker_color="#c7d2fe",
        marker_line_width=0
    ))
    fig6.update_layout(
        title=dict(text="Average Years Experience by Position", x=0.02, xanchor="left", font=dict(size=24, color="#111827")),
        xaxis=dict(title="Average Years", gridcolor="#e5e7eb"),
        yaxis=dict(title="Position", gridcolor="#e5e7eb"),
        template="plotly_white",
        margin=dict(t=60, r=20, b=60, l=120)
    )
    fig6.add_annotation(text="Numbers denote mean years of experience", x=0.02, y=-0.18, showarrow=False, xref='paper', yref='paper', font=dict(color="#6b7280"))
    plots["stacked_position_education"] = pio.to_html(fig6, full_html=False, include_plotlyjs=False)

    # 7) Table: top N candidates (pandas to_html)
    table_html = df.head(100).to_html(classes="table table-striped table-sm", index=False, justify="left")
    plots["table_html"] = table_html

    return plots

def make_insights(df, agg_position, agg_education, agg_experience, pivot_pos_ed):
    insights = {}

    # Pie positions
    if not agg_position.empty:
        top_pos = agg_position.iloc[0]
        insights["pie_positions"] = [
            f"Top position by volume: {top_pos['position_applied']} ({int(top_pos['candidate_count'])} candidates)",
            f"Overall positions covered: {agg_position['position_applied'].nunique()}"
        ]
    else:
        insights["pie_positions"] = ["No position data available"]

    # Bar education
    if not agg_education.empty:
        top_edu = agg_education.sort_values('candidate_count', ascending=False).iloc[0]
        insights["bar_education"] = [
            f"Most common education: {top_edu['education_level']} ({int(top_edu['candidate_count'])})",
            f"Education categories: {agg_education['education_level'].nunique()}"
        ]
    else:
        insights["bar_education"] = ["No education data available"]

    # Combo experience
    if not agg_experience.empty:
        peak_count = agg_experience.sort_values('candidate_count', ascending=False).iloc[0]
        peak_rate = agg_experience.sort_values('suitable_rate', ascending=False).iloc[0]
        insights["combo_experience"] = [
            f"Peak volume at {int(peak_count['years_experience'])} years: {int(peak_count['candidate_count'])}",
            f"Highest suitability at {int(peak_rate['years_experience'])} years: {peak_rate['suitable_rate']:.2f}"
        ]
    else:
        insights["combo_experience"] = ["No experience data available"]

    # Area cumulative
    total = int(df['candidate_id'].nunique())
    insights["area_cumulative"] = [
        f"Total candidates: {total}",
        "Cumulative curve should be non-decreasing across experience"
    ]

    # Heatmap (scatter_individuals key)
    try:
        bins = [0, 2, 5, 8, 11, 20]
        labels = ["0–2", "3–5", "6–8", "9–11", "12+"]
        df_tmp = df.copy()
        df_tmp["years_bin"] = pd.cut(df_tmp["years_experience"], bins=bins, labels=labels, right=True, include_lowest=True)
        hm = df_tmp.pivot_table(index="position_applied", columns="years_bin", values="candidate_id", aggfunc="count", fill_value=0)
        if not hm.empty:
            max_idx = divmod(hm.values.argmax(), hm.shape[1])
            pos = hm.index[max_idx[0]]
            yb = hm.columns[max_idx[1]]
            cnt = int(hm.values[max_idx])
            insights["scatter_individuals"] = [
                f"Densest cell: {pos} at {yb} years bin (count={cnt})",
                "Darker color = higher count"
            ]
        else:
            insights["scatter_individuals"] = ["No heatmap data available"]
    except Exception:
        insights["scatter_individuals"] = ["No heatmap data available"]

    # Position combo
    if not agg_position.empty:
        best_rate = agg_position.sort_values('suitable_rate', ascending=False).iloc[0]
        insights["stacked_position_education"] = [
            f"Top volume: {agg_position.iloc[0]['position_applied']} ({int(agg_position.iloc[0]['candidate_count'])})",
            f"Best suitability: {best_rate['position_applied']} ({best_rate['suitable_rate']:.2f})"
        ]
    else:
        insights["stacked_position_education"] = ["No position data available"]

    return insights

@app.route("/")
def index():
    df, agg_position, agg_education, agg_experience, pivot_pos_ed = load_and_prepare()
    plots = make_plot_divs(df, agg_position, agg_education, agg_experience, pivot_pos_ed)
    insights = make_insights(df, agg_position, agg_education, agg_experience, pivot_pos_ed)
    # Provide summary metrics
    metrics = {
        "total_candidates": int(df["candidate_id"].nunique()),
        "positions": int(df["position_applied"].nunique()),
        "avg_years_experience": round(df["years_experience"].mean(), 2),
        "overall_suitable_rate": round(df["suitable"].mean(), 3),
    }
    return render_template("index.html", plots=plots, metrics=metrics, insights=insights)

if __name__ == "__main__":
    app.run(debug=True)

