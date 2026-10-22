import io
import math
import re
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn is expected to live if current mortality patterns continue.", "nonnegative"),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment; national estimate.", "percent"),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. This is not average wealth or household income.", "nonnegative"),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count.", "nonnegative"),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net number of migrants over the period. It is not an emigration count.", "any"),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicides per 100,000 people.", "nonnegative"),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in consumer prices.", "any"),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP.", "any"),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars.", "nonnegative"),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP.", "any"),
}
WB_API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=200"
BASE_YEAR_NOW = date.today().year
START_YEAR, END_YEAR = BASE_YEAR_NOW - 70, BASE_YEAR_NOW - 1

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("Explore authentic World Bank historical data, fit polynomial models, and investigate functions. Estimates outside the observed years are extrapolations—not forecasts guaranteed to be accurate.")
st.info("Brazil is included as a Latin American comparison country. World Bank indicator coverage varies; the app reports actual available observations instead of promising 70 data points.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code, indicator_code, start_year, end_year):
    url = WB_API.format(country=country_code, indicator=indicator_code, start=start_year, end=end_year)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError("The World Bank API returned no usable observations.")
    rows = []
    for item in payload[1]:
        value = item.get("value")
        year = item.get("date")
        if value is not None and year is not None:
            rows.append({"year": int(year), "value": float(value)})
    frame = pd.DataFrame(rows, columns=["year", "value"])
    if frame.empty:
        raise ValueError("No non-missing observations were returned for this country and indicator.")
    return frame.sort_values("year").drop_duplicates("year").reset_index(drop=True), url

def validate_data(frame, constraint):
    issues = []
    if frame.empty:
        return ["The working table is empty."]
    if frame["year"].isna().any() or not np.all(np.equal(np.mod(frame["year"].dropna(), 1), 0)):
        issues.append("Years must be whole numbers.")
    if frame["year"].duplicated().any():
        issues.append("Years must be unique; remove duplicate years.")
    if frame["value"].isna().any() or not np.isfinite(frame["value"].dropna()).all():
        issues.append("Values must be numeric and finite.")
    if constraint == "nonnegative" and (frame["value"].dropna() < 0).any():
        issues.append("This indicator cannot be negative; check edited values.")
    if constraint == "percent" and ((frame["value"].dropna() < 0) | (frame["value"].dropna() > 100)).any():
        issues.append("Unemployment percentage should be between 0 and 100.")
    return issues

def centered_polyfit(years, values, degree):
    base = int(np.min(years))
    t = np.asarray(years, dtype=float) - base
    coeffs = np.polyfit(t, np.asarray(values, dtype=float), degree)
    predict = lambda year: np.polyval(coeffs, np.asarray(year, dtype=float) - base)
    fitted = predict(years)
    return base, coeffs, predict, fitted

def polynomial_equation(coeffs, base, variable="t"):
    terms = []
    degree = len(coeffs) - 1
    for i, coef in enumerate(coeffs):
        power = degree - i
        if abs(coef) < 1e-12:
            continue
        c = f"{coef:.8g}"
        if power == 0:
            terms.append(c)
        elif power == 1:
            terms.append(f"({c})({variable})")
        else:
            terms.append(f"({c})({variable})^{power}")
    return " + ".join(terms).replace("+ -", "- ") + f", where {variable} = year - {base}"

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

with st.sidebar:
    st.header("Data and model controls")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=0)
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()))
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_increment = st.slider("Graph sample increment (years)", 1, 10, 3)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_names = st.multiselect("Comparison countries", [c for c in COUNTRIES if c != country_name], default=[])
    course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

country_code = COUNTRIES[country_name]
indicator_code, units, definition, constraint = INDICATORS[indicator_name]
try:
    with st.spinner("Retrieving official World Bank data…"):
        source_frame, source_url = fetch_world_bank(country_code, indicator_code, START_YEAR, END_YEAR)
    source_frame = source_frame.rename(columns={"year": "year", "value": "value"})
    source_frame["year"] = source_frame["year"].astype(int)
except Exception as exc:
    st.error(f"Could not retrieve source data: {exc}")
    st.markdown(f"[World Bank indicator page](https://data.worldbank.org/indicator/{indicator_code})")
    st.stop()

if "active_key" not in st.session_state:
    st.session_state.active_key = None
key = (country_code, indicator_code)
if st.session_state.active_key != key:
    st.session_state.source_data = source_frame.copy()
    st.session_state.working_data = source_frame.copy()
    st.session_state.user_modified = False
    st.session_state.active_key = key
    st.session_state.single_result = None
    st.session_state.batch_result = None

st.subheader("Source data")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Available observations", len(source_frame))
m2.metric("Actual year coverage", f"{source_frame.year.min()}–{source_frame.year.max()}")
m3.metric("Indicator", indicator_code)
m4.metric("Retrieved", date.today().isoformat())
st.write(definition)
st.write(f"**Units:** {units}  |  **Source:** [World Bank API]({source_url}) · [Indicator details](https://data.worldbank.org/indicator/{indicator_code})")
st.caption(f"Requested years: {START_YEAR}–{END_YEAR}. Actual coverage and count depend on World Bank reporting.")

left, right = st.columns([3, 1])
with left:
    st.markdown("**Editable working table**")
    st.caption("Edit year/value rows below. Edited values are a user-modified scenario, not official source observations.")
    edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{country_code}_{indicator_code}")
    if not edited.equals(st.session_state.working_data):
        st.session_state.working_data = edited.copy()
        st.session_state.user_modified = True
        st.session_state.single_result = None
        st.session_state.batch_result = None
with right:
    st.markdown("**Data actions**")
    if st.button("Reset to source", use_container_width=True):
        st.session_state.working_data = st.session_state.source_data.copy()
        st.session_state.user_modified = False
        st.session_state.single_result = None
        st.session_state.batch_result = None
        st.rerun()
    if st.session_state.user_modified:
        st.warning("User-modified scenario")
    else:
        st.success("Source values")

working = st.session_state.working_data.copy()
working["year"] = pd.to_numeric(working["year"], errors="coerce")
working["value"] = pd.to_numeric(working["value"], errors="coerce")
issues = validate_data(working, constraint)
if issues:
    for issue in issues:
        st.error(issue)
    st.warning("Fix the working table before fitting a model.")
    st.stop()
working = working.sort_values("year").reset_index(drop=True)
years = working["year"].astype(int).to_numpy()
values = working["value"].astype(float).to_numpy()
if len(np.unique(years)) < degree + 1:
    st.error(f"Degree {degree} requires at least {degree + 1} distinct years; only {len(np.unique(years))} are available.")
    st.stop()

base, coeffs, predict, fitted = centered_polyfit(years, values, degree)
residuals = values - fitted
r2 = r2_score(values, fitted) if len(values) > 1 else float("nan")
mae = mean_absolute_error(values, fitted)
equation = polynomial_equation(coeffs, base)
fit_table = pd.DataFrame({"year": years, "actual": values, "fitted": fitted, "residual": residuals})
with st.expander("Polynomial fit details", expanded=True):
    st.markdown(f"**Model:** `{equation}`")
    st.caption(f"Fit metrics are in-sample. Base year: {base}. Degree: {degree}. Observations: {len(values)}. Historical range: {years.min()}–{years.max()}.")
    a, b, c = st.columns(3)
    a.metric("In-sample R²", f"{r2:.5f}")
    b.metric("In-sample MAE", f"{mae:.5g} {units}")
    c.metric("Historical mean", f"{np.mean(values):.5g} {units}")
    st.dataframe(fit_table, use_container_width=True, hide_index=True)
    if len(values) >= max(12, degree + 4):
        split = int(len(values) * 0.8)
        if split >= degree + 1 and len(values) - split >= 2:
            hold_base, hold_coeffs, _, _ = centered_polyfit(years[:split], values[:split], degree)
            hold_pred = np.polyval(hold_coeffs, years[split:] - hold_base)
            st.write(f"**Chronological holdout MAE:** {mean_absolute_error(values[split:], hold_pred):.5g} {units} (trained on earlier years, tested on later years).")

st.subheader("Visualize the historical fit")
future_end = int(years.max() + horizon)
dense_x = np.linspace(years.min(), max(years.max(), future_end), max(300, (future_end - int(years.min()) + 1) * 4))
dense_y = predict(dense_x)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(years, values, label="Observed / working data", s=24)
ax.plot(dense_x[dense_x <= years.max()], dense_y[dense_x <= years.max()], label="Polynomial fit (historical range)")
if horizon > 0:
    ax.plot(dense_x[dense_x >= years.max()], dense_y[dense_x >= years.max()], linestyle="--", label="Extrapolation")
ax.axvline(years.max(), color="gray", linestyle=":", label="Historical boundary")
sample_years = np.arange(years.min(), years.max() + 1, sample_increment)
ax.scatter(sample_years, predict(sample_years), marker="x", label=f"Model markers every {sample_increment} year(s)")
ax.set_xlabel("Year")
ax.set_ylabel(f"{indicator_name} ({units})")
ax.set_title(f"{country_name}: {indicator_name}")
ax.grid(True, alpha=0.25)
ax.legend()
st.pyplot(fig, use_container_width=True)
st.caption("Polynomial extrapolations can become implausible quickly. No values are silently clipped; residuals and MAE are not prediction intervals.")

st.subheader("Calculate a value")
single_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=int(np.clip(years.max(), 1950, 2100)), step=1)
if st.button("Calculate Value"):
    estimate = float(predict(single_year))
    st.session_state.single_result = {"year": int(single_year), "estimate": estimate, "key": (country_code, indicator_code, degree, tuple(years), tuple(values))}
if st.session_state.single_result:
    result = st.session_state.single_result
    current_key = (country_code, indicator_code, degree, tuple(years), tuple(values))
    if result["key"] == current_key:
        st.write(f"**Year:** {result['year']} · **Estimate:** {result['estimate']:.8g} {units}")
        st.write(f"**Classification:** {'within' if years.min() <= result['year'] <= years.max() else 'outside'} the fitted historical range ({years.min()}–{years.max()}).")
        actual_match = working.loc[working.year == result["year"], "value"]
        if not actual_match.empty:
            st.caption(f"Actual observation: {float(actual_match.iloc[0]):.8g} {units}; the model estimate is a separate fitted value.")
        if abs(result["estimate"]) > max(1e6, 100 * max(1, np.nanmax(np.abs(values)))):
            st.warning("This estimate is unusually large relative to the data. Check extrapolation limits and model assumptions.")

st.markdown("**Batch estimates**")
batch_mode = st.radio("Input method", ["Year range", "Custom years"], horizontal=True)
batch_years = []
if batch_mode == "Year range":
    r1, r2, r3 = st.columns(3)
    batch_start = r1.number_input("Start year", min_value=1950, max_value=2100, value=max(1950, int(years.min())), step=1)
    batch_end = r2.number_input("End year", min_value=1950, max_value=2100, value=min(2100, int(years.max())), step=1)
    batch_step = r3.slider("Range step", 1, 10, 1)
    if batch_end < batch_start:
        st.error("End year must be greater than or equal to start year.")
    else:
        batch_years = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
else:
    custom_years = st.text_input("Comma-separated years", value=f"{int(years.min())}, {int(years.max())}")
    try:
        batch_years = [int(part.strip()) for part in custom_years.split(",") if part.strip()]
        if any(y < 1950 or y > 2100 for y in batch_years):
            st.error("Every requested year must be between 1950 and 2100.")
            batch_years = []
    except ValueError:
        st.error("Enter whole years separated by commas, such as 1980, 1990, 2000.")
        batch_years = []
if st.button("Calculate batch"):
    if not batch_years:
        st.warning("Enter at least one valid year.")
    else:
        st.session_state.batch_result = {
            "key": (country_code, indicator_code, degree, tuple(years), tuple(values)),
            "table": pd.DataFrame({"year": batch_years, "estimate": predict(np.array(batch_years, dtype=float))})
        }
if st.session_state.batch_result:
    br = st.session_state.batch_result
    current_key = (country_code, indicator_code, degree, tuple(years), tuple(values))
    if br["key"] == current_key:
        batch_table = br["table"].copy()
        batch_table["classification"] = np.where(batch_table.year.between(years.min(), years.max()), "within historical range", "outside historical range")
        st.dataframe(batch_table, use_container_width=True, hide_index=True)
    else:
        st.info("Previous batch results were invalidated because the data or model changed.")

if comparison_names:
    st.subheader("Compare countries")
    compare_fig, compare_ax = plt.subplots(figsize=(10, 4.5))
    compare_ax.plot(years, values, marker="o", label=country_name)
    comparison_notes = []
    for cname in comparison_names:
        try:
            cframe, _ = fetch_world_bank(COUNTRIES[cname], indicator_code, START_YEAR, END_YEAR)
            compare_ax.plot(cframe.year, cframe.value, marker=".", label=cname)
            comparison_notes.append(f"{cname}: {cframe.year.min()}–{cframe.year.max()} ({len(cframe)} observations)")
        except Exception as exc:
            comparison_notes.append(f"{cname}: unavailable ({exc})")
    compare_ax.set_xlabel("Year")
    compare_ax.set_ylabel(f"{indicator_name} ({units})")
    compare_ax.grid(True, alpha=0.25)
    compare_ax.legend()
    st.pyplot(compare_fig, use_container_width=True)
    st.caption("Coverage can differ by country; lines may contain gaps where source observations are missing.")
    st.write(" · ".join(comparison_notes))

st.subheader("Function lab")
st.write("Explore a suggested or custom mathematical function. Use `x` as the independent variable. Custom expressions use standard SymPy notation, for example `x**2 - 3*x + 2`, `exp(x)`, `log(x)`, or `sin(x)`.")
context = st.selectbox("Context", ["Business", "Engineering", "Medical", "Sports science", "Biology", "Bioengineering", "Economic forecasting", "Weather", "Custom"])
context_defaults = {
    "Business": ("Revenue model", "x**2 - 4*x + 8", "time", "hours", "revenue", "dollars"),
    "Engineering": ("Projectile-like curve", "-4.9*x**2 + 20*x + 1", "time", "seconds", "height", "meters"),
    "Medical": ("Concentration illustration", "10*exp(-0.3*x)", "time", "hours", "concentration", "mg/L"),
    "Sports science": ("Performance curve", "-0.5*x**2 + 5*x", "time", "minutes", "performance", "score"),
    "Biology": ("Population illustration", "100*exp(0.08*x)", "time", "days", "population", "organisms"),
    "Bioengineering": ("Sensor response illustration", "2*sin(x) + 5", "input", "units", "response", "units"),
    "Economic forecasting": ("Growth illustration", "100*exp(0.03*x)", "time", "years", "index", "index points"),
    "Weather": ("Temperature illustration", "10*sin(2*pi*x/24) + 15", "time", "hours", "temperature", "°C"),
    "Custom": ("Custom function", "x**2", "x", "units", "f(x)", "units"),
}
default_title, default_expr, default_xname, default_xunit, default_yname, default_yunit = context_defaults[context]
desc = st.text_input("Context description", value=default_title)
var1, var2, var3, var4 = st.columns(4)
xname = var1.text_input("Independent variable", value=default_xname)
xunit = var2.text_input("Independent units", value=default_xunit)
yname = var3.text_input("Dependent variable", value=default_yname)
yunit = var4.text_input("Dependent units", value=default_yunit)
function_source = st.radio("Function source", ["Suggested function", "Custom function", "Use historical polynomial"], horizontal=True)
if function_source == "Suggested function":
    expression_text = default_expr
elif function_source == "Use historical polynomial":
    expression_text = None
else:
    expression_text = st.text_input("Enter function in x", value="x**2 - 3*x + 2")
interval_cols = st.columns(3)
x_min = interval_cols[0].number_input("Analysis interval start", value=0.0, step=0.5)
x_max = interval_cols[1].number_input("Analysis interval end", value=10.0, step=0.5)
x_point = interval_cols[2].number_input("Point for tangent", value=2.0, step=0.5)
show_function = st.checkbox("Show function", value=True)
show_first = st.checkbox("Show first derivative", value=True)
show_second = st.checkbox("Show second derivative", value=False)
if x_max <= x_min:
    st.error("Analysis interval end must be greater than start.")
else:
    x_sym = sp.symbols("x", real=True)
    try:
        if function_source == "Use historical polynomial":
            # Translate the historical model to the function-lab variable x = calendar year.
            expr = sum(float(c) * (x_sym - base) ** (len(coeffs) - i - 1) for i, c in enumerate(coeffs))
        else:
            allowed = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
                       "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E, "Abs": sp.Abs}
            expr = sp.sympify(expression_text, locals=allowed)
        if expr.has(sp.Function) and any(not isinstance(node, (sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt, sp.Abs)) for node in expr.atoms(sp.Function)):
            raise ValueError("Unsupported function in expression.")
        first_derivative = sp.diff(expr, x_sym)
        second_derivative = sp.diff(first_derivative, x_sym)
        st.markdown(f"**Function:** `{sp.sstr(expr)}`")
        st.markdown(f"**First derivative:** `{sp.sstr(first_derivative)}`")
        st.markdown(f"**Second derivative:** `{sp.sstr(second_derivative)}`")
        f = sp.lambdify(x_sym, expr, modules=["numpy"])
        df = sp.lambdify(x_sym, first_derivative, modules=["numpy"])
        d2f = sp.lambdify(x_sym, second_derivative, modules=["numpy"])
        xx = np.linspace(float(x_min), float(x_max), 800)
        def safe_eval(fn, arr):
            with np.errstate(all="ignore"):
                out = np.asarray(fn(arr), dtype=float)
                if out.ndim == 0:
                    out = np.full_like(arr, float(out))
                return np.where(np.isfinite(out), out, np.nan)
        yy, dy, d2y = safe_eval(f, xx), safe_eval(df, xx), safe_eval(d2f, xx)
        point_value = float(f(x_point))
        slope = float(df(x_point))
        second_value = float(d2f(x_point))
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        if show_function: ax2.plot(xx, yy, label=f"{yname} = f(x)")
        if show_first: ax2.plot(xx, dy, label="First derivative")
        if show_second: ax2.plot(xx, d2y, label="Second derivative")
        if np.isfinite(point_value): ax2.scatter([x_point], [point_value], zorder=5, label="Selected point")
        if np.isfinite(point_value) and np.isfinite(slope):
            tangent_x = np.linspace(max(x_min, x_point - (x_max-x_min)*0.15), min(x_max, x_point + (x_max-x_min)*0.15), 60)
            ax2.plot(tangent_x, point_value + slope * (tangent_x - x_point), linestyle="--", label="Tangent line")
        ax2.set_xlabel(f"{xname} ({xunit})")
        ax2.set_ylabel(f"{yname} / derivative values (units vary)")
        ax2.set_title(desc)
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        st.pyplot(fig2, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{yname} at x={x_point:g}", f"{point_value:.6g} {yunit}")
        c2.metric("First derivative at point", f"{slope:.6g} {yunit}/{xunit}")
        c3.metric("Second derivative at point", f"{second_value:.6g} {yunit}/{xunit}²")
        avg_rate = (float(f(x_max)) - float(f(x_min))) / (x_max - x_min)
        st.write(f"**Average rate of change** on [{x_min:g}, {x_max:g}]: {avg_rate:.6g} {yunit}/{xunit}.")
        st.write(f"**Context interpretation:** {yname} is the modeled output. The first derivative describes its instantaneous rate of change; the second derivative describes how that rate changes. These interpretations are illustrative and depend on whether the selected function is appropriate for the context.")
        if course in ("Precalculus", "Calculus"):
            st.write("**Changing rates:** Positive derivative values indicate increasing behavior; negative values indicate decreasing behavior. The second derivative helps describe concavity where it is defined.")
        if course == "Calculus":
            st.markdown("**Numerical calculus checks**")
            try:
                critical = sp.solveset(sp.Eq(first_derivative, 0), x_sym, domain=sp.Interval(float(x_min), float(x_max)))
                inflection_candidates = sp.solveset(sp.Eq(second_derivative, 0), x_sym, domain=sp.Interval(float(x_min), float(x_max)))
                st.write(f"Derivative-zero critical candidates on the interval: `{critical}`. Candidates require classification and may not all be extrema.")
                st.write(f"Second-derivative-zero candidates: `{inflection_candidates}`. A zero alone does not prove an inflection; concavity must change.")
                st.caption("Symbolic solving may return conditional or incomplete sets for complicated functions. Treat numerical/implicit results cautiously.")
            except Exception as exc:
                st.warning(f"Could not fully solve critical/inflection candidates: {exc}")
        st.caption("This function lab is educational. It does not establish medical, engineering, or economic validity for the illustrative functions.")
    except Exception as exc:
        st.error(f"Could not parse or analyze this function: {exc}. Check notation, domain restrictions, and the interval.")

st.subheader("Export current work")
export_data = working.copy()
export_data["fitted"] = predict(years)
export_data["residual"] = values - predict(years)
summary = pd.DataFrame([{
    "country": country_name, "country_code": country_code, "indicator": indicator_name,
    "indicator_code": indicator_code, "units": units, "definition": definition,
    "retrieval_date": date.today().isoformat(), "degree": degree, "base_year": base,
    "equation": equation, "in_sample_r2": r2, "in_sample_mae": mae,
    "observation_count": len(working), "historical_start": int(years.min()),
    "historical_end": int(years.max()), "user_modified_scenario": st.session_state.user_modified,
}])
csv_bytes = export_data.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", csv_bytes, file_name="historical_data.csv", mime="text/csv")
xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    export_data.to_excel(writer, sheet_name="Working Data", index=False)
    if st.session_state.batch_result and st.session_state.batch_result["key"] == (country_code, indicator_code, degree, tuple(years), tuple(values)):
        st.session_state.batch_result["table"].to_excel(writer, sheet_name="Calculated Years", index=False)
    elif st.session_state.single_result:
        pd.DataFrame([{"year": st.session_state.single_result["year"], "estimate": st.session_state.single_result["estimate"]}]).to_excel(writer, sheet_name="Calculated Years", index=False)
xlsx_buffer.seek(0)
st.download_button("Download Excel workbook", xlsx_buffer.getvalue(), file_name="historical_data_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

pdf_buffer = io.BytesIO()
pdf_doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter), rightMargin=30, leftMargin=30, topMargin=28, bottomMargin=28)
styles = getSampleStyleSheet()
story = [
    Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
    Paragraph(f"{country_name} — {indicator_name} ({units})", styles["Heading2"]),
    Paragraph(f"Definition: {definition}", styles["BodyText"]),
    Paragraph(f"Source: {source_url}", styles["BodyText"]),
    Paragraph(f"Retrieved: {date.today().isoformat()} · Observations: {len(working)} · Coverage: {years.min()}–{years.max()}", styles["BodyText"]),
    Paragraph(f"Polynomial degree {degree}; base year {base}", styles["BodyText"]),
    Paragraph(f"Equation: {equation}", styles["BodyText"]),
    Paragraph(f"In-sample R²: {r2:.6g}; in-sample MAE: {mae:.6g}; mean: {np.mean(values):.6g}", styles["BodyText"]),
    Paragraph(f"Edited scenario: {'Yes' if st.session_state.user_modified else 'No'}", styles["BodyText"]),
    Spacer(1, 10),
]
pdf_fig, pdf_ax = plt.subplots(figsize=(9, 3.5))
pdf_ax.scatter(years, values, label="Observed / working data", s=12)
pdf_ax.plot(dense_x, dense_y, label="Polynomial model")
pdf_ax.set_xlabel("Year"); pdf_ax.set_ylabel(f"{indicator_name} ({units})"); pdf_ax.grid(True, alpha=0.2); pdf_ax.legend()
story.append(Image(fig_to_png(pdf_fig), width=620, height=240))
preview = [["Year", "Value", "Fitted", "Residual"]]
for row in fit_table.head(18).itertuples(index=False):
    preview.append([str(int(row.year)), f"{row.actual:.5g}", f"{row.fitted:.5g}", f"{row.residual:.5g}"])
table = Table(preview, repeatRows=1)
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.extend([Spacer(1, 8), Paragraph("Data preview (first 18 rows)", styles["Heading3"]), table,
              Spacer(1, 8), Paragraph("Limitations: Polynomial fits describe the selected data but do not establish causation. Extrapolations may be implausible. MAE and residuals are not 95% prediction intervals.", styles["BodyText"])])
pdf_doc.build(story)
pdf_buffer.seek(0)
st.download_button("Download PDF report", pdf_buffer.getvalue(), file_name="historical_data_report.pdf", mime="application/pdf")

with st.expander("How the app works and deployment"):
    st.markdown("""
**Data flow:** The app requests official World Bank indicator data, caches the response for one hour, keeps a source copy, and exposes a separate editable working table. Reset restores the downloaded source values.

**Model:** The polynomial is fit using centered time, `t = year - base year`, to improve numerical stability. R² and MAE describe in-sample fit only. The optional chronological holdout score uses earlier years for fitting and later years for testing.

**Limitations:** Historical data can be missing. Polynomial models can overfit and extrapolate poorly. The function lab's contextual examples are teaching illustrations, not validated forecasts or professional advice. Symbolic critical-point solving can be incomplete for complicated functions.

**Run locally:**
1. `pip install -r requirements.txt`
2. `python -m streamlit run app.py`

**Deploy:** Upload `app.py` and `requirements.txt` to a GitHub repository. In Streamlit Community Cloud, create an app from that repository and choose `app.py` as the entry point.

**Render:** Create a Python Web Service. Build command: `pip install -r requirements.txt`. Start command: `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`.

**Checks:** Verify the source URL, edit and reset data, try duplicate years and insufficient observations, compare single and batch predictions, change each control, test invalid expressions such as `log(-1)`, test an unavailable indicator/country response, and download all three export formats. For API or deployment errors, inspect the app logs and confirm outbound access to `api.worldbank.org`.
""")
