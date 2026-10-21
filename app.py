import html
import random

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Hello World Motion Lab",
    page_icon="✨",
    layout="wide",
)

st.title("✨ Hello World Motion Lab")
st.write("Type your name, choose a style, and add a little motion.")

FONTS = {
    "Arial": "Arial, Helvetica, sans-serif",
    "Courier New": "'Courier New', monospace",
    "Georgia": "Georgia, serif",
    "Times New Roman": "'Times New Roman', serif",
    "Verdana": "Verdana, sans-serif",
    "Trebuchet MS": "'Trebuchet MS', sans-serif",
    "Comic Sans MS": "'Comic Sans MS', cursive",
    "Impact": "Impact, sans-serif",
}

BACKGROUNDS = {
    "Light": ("#f8fafc", "#172033"),
    "Dark": ("#111827", "#f9fafb"),
    "Ocean": ("linear-gradient(135deg, #0f766e, #1d4ed8)", "#ffffff"),
    "Sunset": ("linear-gradient(135deg, #7c2d12, #db2777, #7c3aed)", "#ffffff"),
    "Candy": ("linear-gradient(135deg, #fbcfe8, #c4b5fd, #bfdbfe)", "#3b0764"),
    "Forest": ("linear-gradient(135deg, #14532d, #166534, #365314)", "#f0fdf4"),
    "Midnight": ("radial-gradient(circle at top, #312e81, #020617 75%)", "#e0e7ff"),
}

EFFECTS = ["None", "Rain", "Snow", "Hail", "Wind gust", "Confetti", "Tornado"]

with st.sidebar:
    st.header("Customize your name")
    name = st.text_input("Your name or message", value="Hello World", max_chars=80)
    font_name = st.selectbox("Font type", list(FONTS.keys()))
    font_size = st.slider("Font size", min_value=20, max_value=100, value=52, step=2)
    background_name = st.selectbox("Background", list(BACKGROUNDS.keys()))
    display_mode = st.selectbox("Name display mode", ["Centered", "Ticker tape", "Repeated"])
    effect = st.selectbox("Dynamic action", EFFECTS)
    speed = st.slider("Animation speed", min_value=1, max_value=10, value=5,
                      help="1 is slow; 10 is fast.")
    st.caption("The motion runs in your browser, so it loops continuously without repeatedly rerunning Streamlit.")

bg, text_color = BACKGROUNDS[background_name]
safe_name = html.escape(name.strip() or "Hello World")
font_css = FONTS[font_name]
# Keep the selected background value and name content safe for CSS/HTML.
safe_bg = bg.replace("'", "").replace(";", "")
safe_text_color = text_color.replace(";", "")
duration = 11 - speed  # 1..10 maps to 10..1 seconds

random.seed(21)
particles = []
symbols = {
    "Rain": "💧",
    "Snow": "❄",
    "Hail": "🧊",
    "Wind gust": "〰",
    "Confetti": "🎉",
    "Tornado": "🌪",
}
symbol = symbols.get(effect, "")

if effect != "None":
    count = 42 if effect in ("Rain", "Snow", "Hail", "Confetti") else 18
    for i in range(count):
        left = random.randint(0, 100)
        delay = random.uniform(0, duration)
        size = random.randint(12, 25) if effect != "Tornado" else random.randint(18, 32)
        drift = random.randint(-100, 100)
        fall_duration = random.uniform(max(1.2, duration * 0.55), duration * 1.5)
        particles.append(
            f'<span class="particle p-{i}" style="left:{left}%;'
            f'animation-delay:-{delay:.2f}s;animation-duration:{fall_duration:.2f}s;'
            f'font-size:{size}px;--drift:{drift}px">{symbol}</span>'
        )

if display_mode == "Repeated":
    name_markup = '<div class="repeated">' + "".join(
        f'<div class="name-line">{safe_name}</div>' for _ in range(5)
    ) + "</div>"
elif display_mode == "Ticker tape":
    name_markup = (
        '<div class="ticker"><div class="ticker-track">'
        f'<span>{safe_name}</span><span>{safe_name}</span>'
        f'<span>{safe_name}</span><span>{safe_name}</span>'
        "</div></div>"
    )
else:
    name_markup = f'<div class="center-name">{safe_name}</div>'

effect_class = {
    "Rain": "rain",
    "Snow": "snow",
    "Hail": "hail",
    "Wind gust": "wind",
    "Confetti": "confetti",
    "Tornado": "tornado",
}.get(effect, "no-effect")

# CSS animations run continuously inside the embedded browser component.
html_doc = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; }}
.stage {{
  position: relative; height: 590px; width: 100%; overflow: hidden;
  border-radius: 18px; background: {safe_bg}; color: {safe_text_color};
  font-family: {font_css}; isolation: isolate;
}}
.effect-layer {{ position:absolute; inset:0; overflow:hidden; pointer-events:none; z-index:1; }}
.particle {{ position:absolute; top:-55px; opacity:.88; user-select:none; }}
.rain .particle, .snow .particle, .hail .particle, .confetti .particle {{
  animation-name: fall; animation-timing-function: linear; animation-iteration-count: infinite;
}}
.rain .particle {{ filter: hue-rotate(150deg); }}
.snow .particle {{ animation-name: float-fall; }}
.hail .particle {{ animation-name: hail-fall; }}
.confetti .particle {{ animation-name: confetti-fall; }}
.wind .particle {{ top: auto; left:-15% !important; animation: gust {duration:.2f}s ease-in-out infinite; }}
.tornado .particle {{
  top: 50%; left: 50% !important; animation: spin {max(1.2, duration/2):.2f}s linear infinite;
}}
@keyframes fall {{
  from {{ transform: translate3d(0,-50px,0); }}
  to {{ transform: translate3d(var(--drift),680px,0); }}
}}
@keyframes float-fall {{
  0% {{ transform: translate3d(0,-50px,0) rotate(0deg); }}
  50% {{ transform: translate3d(calc(var(--drift) * -0.5),300px,0) rotate(180deg); }}
  100% {{ transform: translate3d(var(--drift),680px,0) rotate(360deg); }}
}}
@keyframes hail-fall {{
  0% {{ transform: translate3d(0,-50px,0); }}
  70% {{ transform: translate3d(var(--drift),450px,0); }}
  80% {{ transform: translate3d(calc(var(--drift) * .7),420px,0); }}
  100% {{ transform: translate3d(var(--drift),680px,0); }}
}}
@keyframes confetti-fall {{
  from {{ transform: translate3d(0,-50px,0) rotate(0deg); }}
  to {{ transform: translate3d(var(--drift),680px,0) rotate(900deg); }}
}}
@keyframes gust {{
  0% {{ transform: translateX(-10vw) translateY(0) rotate(0deg); opacity:0; }}
  20% {{ opacity:.85; }}
  50% {{ transform: translateX(60vw) translateY(90px) rotate(12deg); }}
  100% {{ transform: translateX(125vw) translateY(-25px) rotate(-8deg); opacity:0; }}
}}
@keyframes spin {{
  0% {{ transform: translate(-50%,-50%) rotate(0deg) translateX(20px) scale(.6); opacity:.25; }}
  50% {{ opacity:.95; }}
  100% {{ transform: translate(-50%,-50%) rotate(720deg) translateX(150px) scale(1.5); opacity:.1; }}
}}
.name-layer {{
  position:absolute; inset:0; z-index:5; display:flex; align-items:center;
  justify-content:center; padding:28px; overflow:hidden;
  font-size:{font_size}px; font-weight:700; text-shadow:0 2px 10px rgba(0,0,0,.16);
}}
.center-name {{ text-align:center; overflow-wrap:anywhere; max-width:100%; }}
.repeated {{ width:100%; text-align:center; line-height:1.35; overflow-wrap:anywhere; }}
.ticker {{ width:100%; overflow:hidden; white-space:nowrap; }}
.ticker-track {{ display:flex; width:max-content; animation: marquee {max(5, 16-speed):.2f}s linear infinite; }}
.ticker-track span {{ padding-right:1.5em; }}
@keyframes marquee {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
.credit {{
  position:absolute; z-index:6; bottom:12px; left:0; right:0; text-align:center;
  font:12px Arial,sans-serif; opacity:.65;
}}
@media (prefers-reduced-motion: reduce) {{
  .particle, .ticker-track {{ animation-duration: 30s !important; }}
}}
</style>
</head>
<body>
<div class="stage">
  <div class="effect-layer {effect_class}">{''.join(particles)}</div>
  <div class="name-layer">{name_markup}</div>
  <div class="credit">Hello World Motion Lab · {html.escape(effect)}</div>
</div>
</body>
</html>
"""

components.html(html_doc, height=610, scrolling=False)
st.caption("Tip: Try different fonts, backgrounds, and effects. Your selections update the preview above.")
