from flask import Flask, request, render_template_string

app = Flask(__name__)

# simple HTML template inside Python
html_template = """
<!doctype html>
<html>
  <head>
    <title>AI vs AI Detector</title>
  </head>
  <body>
    <h1>AI vs AI: Detecting Machine-Generated Text</h1>
    <form method="POST">
      <textarea name="user_text" rows="10" cols="60" placeholder="Paste text here..."></textarea><br><br>
      <button type="submit">Check</button>
    </form>
    {% if result %}
      <h2>Result: {{ result }}</h2>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        text = request.form.get("user_text")

        # 🔎 Very basic fake detection logic for now
        if "AI" in text or "ChatGPT" in text:
            result = "Likely AI-generated 🤖"
        else:
            result = "Likely Human-written 👤"

    return render_template_string(html_template, result=result)

if __name__ == "__main__":
    app.run(debug=True)
