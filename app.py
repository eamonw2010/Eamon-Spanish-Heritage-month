import io
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average years a newborn is expected to live if current mortality patterns continue.", "nonnegative"),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment; national estimate.", "percent"),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "GDP divided by midyear population; not average wealth or household income.", "nonnegative"),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; a rate, not a birth count.", "nonnegative"),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net number of migrants over the period; not an emigration count.", "any"),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people.", "nonnegative"),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in consumer prices.", "any"),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP.", "any"),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars.", "nonnegative"),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.", "any"),
}
CURRENT_YEAR = date.today().year
START_YEAR, END_YEAR = CURRENT_YEAR - 70, CURRENT_YEAR - 1
API_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(country, indicator, start_year, end_year):
    url = API_URL.format(country=country, indicator=indicator, start=start_year, end=end_year)
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError("The World Bank API returned no usable data.")
    rows = []
    for item in payload[1]:
        if item.get("value") is not None and item.get("date") is not None:
            rows.append({"year": int(item["date"]), "value": float(item["value"])})
    frame = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year")
    frame = frame.drop_duplicates("year").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No non-missing observations are available for this selection.")
    return frame, url

def validate(frame, constraint):
    errors = []
    if frame.empty:
        return ["The working table is empty."]
    years = pd.to_numeric(frame["year"], errors="coerce")
    values = pd.to_numeric(frame["value"], errors="coerce")
    if years.isna().any() or not np.all(np.equal(np.mod(years.dropna(), 1), 0)):
        errors.append("Years must be whole numbers.")
    if years.duplicated().any():
        errors.append("Years must be unique; remove duplicate years.")
    if values.isna().any() or not np.isfinite(values.dropna()).all():
        errors.append("Values must be numeric and finite.")
    if constraint == "nonnegative" and (values.dropna() < 0).any():
        errors.append("This indicator cannot be negative.")
    if constraint == "percent" and ((values.dropna() < 0) | (values.dropna() > 100)).any():
        errors.append("Unemployment percentage must be between 0 and 100.")
    return errors

def fit_poly(years, values, degree):
    base = int(np.min(years))
    coefficients = np.polyfit(np.asarray(years, float) - base, np.asarray(values, float), degree)
    return base, coefficients

def predict(coefficients, base, years):
    return np.polyval(coefficients, np.asarray(years, dtype=float) - base)

def equation_text(coefficients, base):
    degree = len(coefficients) - 1
    parts = []
    for i, coefficient in enumerate(coefficients):
        power = degree - i
        if abs(coefficient) < 1e-12:
            continue
        c = f"{coefficient:.8g}"
        if power == 0:
            parts.append(c)
        elif power == 1:
            parts.append(f"({c})t")
        else:
            parts.append(f"({c})t^{power}")
    return " + ".join(parts).replace("+ -", "- ") + f", where t = year - {base}"

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("Official World Bank historical data, editable tables, polynomial models, predictions, and a mathematical function lab.")
st.info("Brazil is included as a Latin American comparison. Data availability varies by indicator and country, so the app reports actual coverage.")

with st.sidebar:
    st.header("Controls")
    country_name = st.selectbox("Country", list(COUNTRIES))
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 3)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_names = st.multiselect("Comparison countries", [name for name in COUNTRIES if name != country_name])
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    printer_mode = st.checkbox("Printer-friendly view")

country_code = COUNTRIES[country_name]
indicator_code, units, definition, constraint = INDICATORS[indicator_name]
try:
    with st.spinner("Loading World Bank data…"):
        source, source_url = fetch_data(country_code, indicator_code, START_YEAR, END_YEAR)
except Exception as exc:
    st.error(f"Could not load World Bank data: {exc}")
    st.markdown(f"[Open the World Bank indicator page](https://data.worldbank.org/indicator/{indicator_code})")
    st.stop()

selection_key = f"{country_code}_{indicator_code}"
if st.session_state.get("selection_key") != selection_key:
    st.session_state.selection_key = selection_key
    st.session_state.source_frame = source.copy()
    st.session_state.working_frame = source.copy()
    st.session_state.user_modified = False
    st.session_state.single_result = None
    st.session_state.batch_result = None

st.subheader("Source data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", len(source))
c2.metric("Year coverage", f"{source.year.min()}–{source.year.max()}")
c3.metric("Indicator code", indicator_code)
c4.metric("Retrieved", date.today().isoformat())
st.write(definition)
st.write(f"**Units:** {units} · **Source:** [World Bank API]({source_url}) · [Indicator details](https://data.worldbank.org/indicator/{indicator_code})")
st.caption(f"Requested years: {START_YEAR}–{END_YEAR}. Actual coverage depends on source availability.")

st.markdown("### Editable working data")
st.caption("Edits create a user-modified scenario. Use Reset to source to restore the original retrieved observations.")
edited_frame = st.data_editor(st.session_state.working_frame, num_rows="dynamic", use_container_width=True, key=f"editor_{selection_key}")
if not edited_frame.equals(st.session_state.working_frame):
    st.session_state.working_frame = edited_frame.copy()
    st.session_state.user_modified = True
    st.session_state.single_result = None
    st.session_state.batch_result = None
if st.button("Reset to source"):
    st.session_state.working_frame = st.session_state.source_frame.copy()
    st.session_state.user_modified = False
    st.session_state.single_result = None
    st.session_state.batch_result = None
    st.rerun()
st.warning("User-modified scenario") if st.session_state.user_modified else st.success("Using source values")

working = st.session_state.working_frame.copy()
working["year"] = pd.to_numeric(working["year"], errors="coerce")
working["value"] = pd.to_numeric(working["value"], errors="coerce")
errors = validate(working, constraint)
if errors:
    for error in errors:
        st.error(error)
    st.stop()
working = working.sort_values("year").reset_index(drop=True)
years = working.year.astype(int).to_numpy()
values = working.value.astype(float).to_numpy()
if len(np.unique(years)) < degree + 1:
    st.error(f"Degree {degree} requires at least {degree + 1} distinct years. Only {len(np.unique(years))} are available.")
    st.stop()

base, coefficients = fit_poly(years, values, degree)
fitted = predict(coefficients, base, years)
residuals = values - fitted
sst = float(np.sum((values - np.mean(values)) ** 2))
r_squared = 1 - float(np.sum(residuals ** 2)) / sst if sst > 0 else float("nan")
mae = float(np.mean(np.abs(residuals)))
equation = equation_text(coefficients, base)
fit_table = pd.DataFrame({"year": years, "actual": values, "fitted": fitted, "residual": residuals})

st.subheader("Polynomial fit")
st.code(equation)
st.caption(f"In-sample metrics · Base year: {base} · Degree: {degree} · Observations: {len(years)} · Historical range: {years.min()}–{years.max()}")
m1, m2, m3 = st.columns(3)
m1.metric("In-sample R²", f"{r_squared:.5f}")
m2.metric("In-sample MAE", f"{mae:.5g} {units}")
m3.metric("Historical mean", f"{np.mean(values):.5g} {units}")
st.dataframe(fit_table, use_container_width=True, hide_index=True)

if len(years) >= max(12, degree + 4):
    split = int(len(years) * 0.8)
    if split >= degree + 1 and len(years) - split >= 2:
        hb, hc = fit_poly(years[:split], values[:split], degree)
        holdout_mae = float(np.mean(np.abs(values[split:] - predict(hc, hb, years[split:]))))
        st.write(f"Chronological holdout MAE: **{holdout_mae:.5g} {units}** (trained on earlier observations, tested on later observations).")

st.subheader("Historical data and fitted curve")
future_end = int(years.max() + horizon)
dense_years = np.linspace(years.min(), max(years.max(), future_end), min(1000, max(200, (future_end - int(years.min()) + 1) * 4)))
plot_frame = pd.DataFrame({"year": dense_years, "Polynomial model": predict(coefficients, base, dense_years)})
plot_frame["Historical fit"] = np.where(plot_frame.year <= years.max(), plot_frame["Polynomial model"], np.nan)
plot_frame["Extrapolation"] = np.where(plot_frame.year >= years.max(), plot_frame["Polynomial model"], np.nan)
observed_frame = pd.DataFrame({"year": years, "Observed / working data": values}).set_index("year")
st.line_chart(plot_frame.set_index("year")[["Historical fit", "Extrapolation"]], height=350)
st.scatter_chart(observed_frame, x=None, y="Observed / working data", height=250)
sample_years = np.arange(int(years.min()), int(years.max()) + 1, sample_increment)
st.caption(f"Model sample increment: {sample_increment} year(s). Sample model values:")
st.dataframe(pd.DataFrame({"year": sample_years, "model_value": predict(coefficients, base, sample_years)}), use_container_width=True, hide_index=True)
st.caption("The dashed conceptual boundary is the last observed year. Polynomial extrapolations can be implausible; MAE is not a prediction interval.")

st.subheader("Calculate a value")
single_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=int(np.clip(years.max(), 1950, 2100)), step=1)
if st.button("Calculate Value"):
    st.session_state.single_result = {
        "key": (country_code, indicator_code, degree, tuple(years), tuple(values)),
        "year": int(single_year), "estimate": float(predict(coefficients, base, [single_year])[0])
    }
current_model_key = (country_code, indicator_code, degree, tuple(years), tuple(values))
result = st.session_state.single_result
if result and result["key"] == current_model_key:
    st.write(f"**Estimate for {result['year']}:** {result['estimate']:.8g} {units}")
    st.write(f"**Range:** {'within' if years.min() <= result['year'] <= years.max() else 'outside'} the fitted historical range ({years.min()}–{years.max()}).")
    actual = working.loc[working.year == result["year"], "value"]
    if not actual.empty:
        st.caption(f"Actual observation: {float(actual.iloc[0]):.8g} {units}. The model estimate is separate from the actual value.")
    if abs(result["estimate"]) > max(1e6, 100 * max(1, float(np.max(np.abs(values))))):
        st.warning("This estimate is unusually large compared with the data. Check extrapolation and model assumptions.")

st.markdown("### Batch estimates")
batch_mode = st.radio("Batch input method", ["Year range", "Custom years"], horizontal=True)
batch_years = []
if batch_mode == "Year range":
    a, b, c = st.columns(3)
    start_batch = a.number_input("Start year", min_value=1950, max_value=2100, value=max(1950, int(years.min())), step=1)
    end_batch = b.number_input("End year", min_value=1950, max_value=2100, value=min(2100, int(years.max())), step=1)
    step_batch = c.slider("Range step", 1, 10, 1)
    if end_batch < start_batch:
        st.error("End year must be greater than or equal to start year.")
    else:
        batch_years = list(range(int(start_batch), int(end_batch) + 1, int(step_batch)))
else:
    custom = st.text_input("Comma-separated years", value=f"{int(years.min())}, {int(years.max())}")
    try:
        batch_years = [int(part.strip()) for part in custom.split(",") if part.strip()]
        if any(y < 1950 or y > 2100 for y in batch_years):
            st.error("Every year must be between 1950 and 2100.")
            batch_years = []
    except ValueError:
        st.error("Enter whole years separated by commas, such as 1980, 1990, 2000.")
        batch_years = []
if st.button("Calculate batch"):
    if not batch_years:
        st.warning("Enter at least one valid year.")
    else:
        st.session_state.batch_result = {
            "key": current_model_key,
            "table": pd.DataFrame({"year": batch_years, "estimate": predict(coefficients, base, batch_years)})
        }
batch_result = st.session_state.batch_result
if batch_result and batch_result["key"] == current_model_key:
    batch_table = batch_result["table"].copy()
    batch_table["classification"] = np.where(batch_table.year.between(years.min(), years.max()), "within historical range", "outside historical range")
    st.dataframe(batch_table, use_container_width=True, hide_index=True)

if comparison_names:
    st.subheader("Country comparison")
    comparison_series = {}
    notes = []
    for name in comparison_names:
        try:
            comp, _ = fetch_data(COUNTRIES[name], indicator_code, START_YEAR, END_YEAR)
            comparison_series[name] = comp.set_index("year")["value"]
            notes.append(f"{name}: {comp.year.min()}–{comp.year.max()} ({len(comp)} observations)")
        except Exception as exc:
            notes.append(f"{name}: unavailable ({exc})")
    comparison_series[country_name] = working.set_index("year")["value"]
    compare_frame = pd.DataFrame(comparison_series).sort_index()
    st.line_chart(compare_frame, height=350)
    st.caption("Coverage differs by country. Missing years appear as gaps rather than invented values.")
    st.write(" · ".join(notes))

st.subheader("Function lab")
st.write("Use `x` as the independent variable. Custom examples: `x**2 - 3*x + 2`, `exp(x)`, `log(x)`, `sin(x)`.")
contexts = {
    "Business": ("Revenue illustration", "x**2 - 4*x + 8", "time", "hours", "revenue", "dollars"),
    "Engineering": ("Projectile-like illustration", "-4.9*x**2 + 20*x + 1", "time", "seconds", "height", "meters"),
    "Medical": ("Concentration illustration", "10*exp(-0.3*x)", "time", "hours", "concentration", "mg/L"),
    "Sports science": ("Performance illustration", "-0.5*x**2 + 5*x", "time", "minutes", "performance", "score"),
    "Biology": ("Population illustration", "100*exp(0.08*x)", "time", "days", "population", "organisms"),
    "Bioengineering": ("Sensor response illustration", "2*sin(x) + 5", "input", "units", "response", "units"),
    "Economic forecasting": ("Growth illustration", "100*exp(0.03*x)", "time", "years", "index", "index points"),
    "Weather": ("Temperature illustration", "10*sin(2*pi*x/24) + 15", "time", "hours", "temperature", "°C"),
    "Custom": ("Custom function", "x**2", "x", "units", "f(x)", "units"),
}
context = st.selectbox("Context", list(contexts))
default_desc, default_expr, default_xname, default_xunit, default_yname, default_yunit = contexts[context]
description = st.text_input("Context description", default_desc)
cols = st.columns(4)
x_name = cols[0].text_input("Independent variable", default_xname)
x_unit = cols[1].text_input("Independent units", default_xunit)
y_name = cols[2].text_input("Dependent variable", default_yname)
y_unit = cols[3].text_input("Dependent units", default_yunit)
function_source = st.radio("Function source", ["Suggested function", "Custom function", "Historical polynomial"], horizontal=True)
if function_source == "Suggested function":
    expression_text = default_expr
elif function_source == "Custom function":
    expression_text = st.text_input("Function expression", "x**2 - 3*x + 2")
else:
    expression_text = None
c1, c2, c3 = st.columns(3)
x_min = c1.number_input("Interval start", value=0.0, step=0.5)
x_max = c2.number_input("Interval end", value=10.0, step=0.5)
x_point = c3.number_input("Point for tangent", value=2.0, step=0.5)
show_function = st.checkbox("Show function", True)
show_first = st.checkbox("Show first derivative", True)
show_second = st.checkbox("Show second derivative", False)

if x_max <= x_min:
    st.error("Interval end must be greater than interval start.")
else:
    try:
        x = sp.symbols("x", real=True)
        if function_source == "Historical polynomial":
            expression = sum(float(coef) * (x - base) ** (len(coefficients) - i - 1) for i, coef in enumerate(coefficients))
        else:
            allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
                       "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "Abs": sp.Abs}
            expression = sp.sympify(expression_text, locals=allowed)
        first_derivative = sp.diff(expression, x)
        second_derivative = sp.diff(first_derivative, x)
        st.write(f"**f(x):** `{sp.sstr(expression)}`")
        st.write(f"**f'(x):** `{sp.sstr(first_derivative)}`")
        st.write(f"**f''(x):** `{sp.sstr(second_derivative)}`")
        fn = sp.lambdify(x, expression, "numpy")
        dfn = sp.lambdify(x, first_derivative, "numpy")
        d2fn = sp.lambdify(x, second_derivative, "numpy")
        xx = np.linspace(x_min, x_max, 400)
        def safe_values(function, input_values):
            with np.errstate(all="ignore"):
                values_out = np.asarray(function(input_values), dtype=float)
                if values_out.ndim == 0:
                    values_out = np.full_like(input_values, float(values_out))
                return np.where(np.isfinite(values_out), values_out, np.nan)
        f_values = safe_values(fn, xx)
        first_values = safe_values(dfn, xx)
        second_values = safe_values(d2fn, xx)
        chart = pd.DataFrame({"x": xx})
        if show_function:
            chart[y_name] = f_values
        if show_first:
            chart["First derivative"] = first_values
        if show_second:
            chart["Second derivative"] = second_values
        st.line_chart(chart.set_index("x"), height=350)
        point_value = float(fn(x_point))
        slope = float(dfn(x_point))
        second_value = float(d2fn(x_point))
        d1, d2, d3 = st.columns(3)
        d1.metric(f"{y_name} at x={x_point:g}", f"{point_value:.6g} {y_unit}")
        d2.metric("First derivative", f"{slope:.6g} {y_unit}/{x_unit}")
        d3.metric("Second derivative", f"{second_value:.6g} {y_unit}/{x_unit}²")
        average_rate = (float(fn(x_max)) - float(fn(x_min))) / (x_max - x_min)
        st.write(f"**Average rate of change** on [{x_min:g}, {x_max:g}]: {average_rate:.6g} {y_unit}/{x_unit}.")
        st.write(f"**Interpretation:** {y_name} is the modeled output. The first derivative gives its instantaneous rate of change, and the second derivative describes how that rate changes. These example functions are illustrations, not validated professional models.")
        if course in ("Precalculus", "Calculus"):
            st.write("Positive first derivative values indicate increasing behavior; negative values indicate decreasing behavior. The second derivative helps describe concavity where defined.")
        if course == "Calculus":
            critical = sp.solveset(sp.Eq(first_derivative, 0), x, domain=sp.Interval(float(x_min), float(x_max)))
            inflection_candidates = sp.solveset(sp.Eq(second_derivative, 0), x, domain=sp.Interval(float(x_min), float(x_max)))
            st.write(f"Critical-point candidates where f'(x)=0: `{critical}`. Each candidate needs classification.")
            st.write(f"Inflection candidates where f''(x)=0: `{inflection_candidates}`. Confirm a concavity change before calling a point an inflection.")
    except Exception as exc:
        st.error(f"Could not analyze this function: {exc}. Check the expression and domain.")

st.subheader("Download reports")
export_frame = working.copy()
export_frame["fitted"] = fitted
export_frame["residual"] = residuals
summary = pd.DataFrame([{
    "country": country_name, "indicator": indicator_name, "indicator_code": indicator_code,
    "units": units, "definition": definition, "retrieval_date": date.today().isoformat(),
    "degree": degree, "base_year": base, "equation": equation, "in_sample_r2": r_squared,
    "in_sample_mae": mae, "observations": len(years), "historical_start": int(years.min()),
    "historical_end": int(years.max()), "user_modified_scenario": st.session_state.user_modified
}])
st.download_button("Download CSV", export_frame.to_csv(index=False).encode("utf-8"), "historical_data.csv", "text/csv")

excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    export_frame.to_excel(writer, sheet_name="Working Data", index=False)
    if batch_result and batch_result["key"] == current_model_key:
        batch_result["table"].to_excel(writer, sheet_name="Calculated Years", index=False)
    elif result and result["key"] == current_model_key:
        pd.DataFrame([{"year": result["year"], "estimate": result["estimate"]}]).to_excel(writer, sheet_name="Calculated Years", index=False)
st.download_button("Download Excel workbook", excel_buffer.getvalue(), "historical_data_explorer.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_buffer = io.BytesIO()
document = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=24, bottomMargin=24)
styles = getSampleStyleSheet()
story = [
    Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
    Paragraph(f"{country_name} — {indicator_name} ({units})", styles["Heading2"]),
    Paragraph(definition, styles["BodyText"]),
    Paragraph(f"Source: {source_url}", styles["BodyText"]),
    Paragraph(f"Retrieved {date.today().isoformat()} · {len(years)} observations · coverage {years.min()}–{years.max()}", styles["BodyText"]),
    Paragraph(f"Degree {degree}; base year {base}; equation: {equation}", styles["BodyText"]),
    Paragraph(f"In-sample R²: {r_squared:.6g}; in-sample MAE: {mae:.6g}; mean: {np.mean(values):.6g}", styles["BodyText"]),
    Paragraph(f"User-modified scenario: {st.session_state.user_modified}", styles["BodyText"]),
    Spacer(1, 10),
]
preview = [["Year", "Value", "Fitted", "Residual"]]
for row in fit_table.head(25).itertuples(index=False):
    preview.append([str(int(row.year)), f"{row.actual:.5g}", f"{row.fitted:.5g}", f"{row.residual:.5g}"])
table = Table(preview, repeatRows=1)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
]))
story += [table, Spacer(1, 10), Paragraph("Limitations: Polynomial models may overfit and extrapolate poorly. Fit metrics describe the supplied data and are not prediction intervals. Function-lab examples are educational illustrations.", styles["BodyText"])]
document.build(story)
st.download_button("Download PDF report", pdf_buffer.getvalue(), "historical_data_report.pdf", "application/pdf")

with st.expander("Run and deploy instructions"):
    st.markdown("""
**Run locally**
1. `pip install -r requirements.txt`
2. `python -m streamlit run app.py`

**Streamlit Community Cloud:** Upload `app.py` and `requirements.txt` to the same folder in GitHub. Create an app from that repository and set the main file to `app.py`.

**Render:** Build command: `pip install -r requirements.txt`  
Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`

**Checks:** Try changing country and indicator, editing/resetting data, entering duplicate years, fitting different polynomial degrees, comparing single and batch estimates, changing each function-lab control, and downloading CSV/Excel/PDF. If deployment fails, check Render logs and verify that `requirements.txt` is in the repository's root directory.
""")
