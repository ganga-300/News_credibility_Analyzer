from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app2 import analyze_news

app = Flask(__name__)
CORS(app)

@app.route("/")
def health():
    return jsonify({"status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    result = analyze_news(text)
    raw_output = result.get("final_output", "")
    lines = raw_output.split("\n")

    summary, analysis = "", ""
    risk_list = []
    verdict_line, confidence_line = "", ""
    section = None

    for line in lines:
        line = line.strip()
        if line == "SUMMARY": section = "summary"
        elif line == "ANALYSIS": section = "analysis"
        elif line == "RISK_FACTORS": section = "risks"
        elif line == "VERDICT": section = "verdict"
        elif line == "DISCLAIMER": section = None
        elif section == "summary" and line: summary += line + " "
        elif section == "analysis" and line: analysis += line + " "
        elif section == "risks" and line.startswith("-"): risk_list.append(line[1:].strip())
        elif section == "verdict":
            if line.startswith("Credibility:"): verdict_line = line.replace("Credibility:", "").strip()
            if line.startswith("Confidence:"): confidence_line = line.replace("Confidence:", "").replace("%", "").strip()

    ml_pred = result.get("ml_prediction", -1)
    ml_conf = result.get("ml_confidence", 0.0)
    verdict_map = {"HIGH": "MOSTLY-TRUE", "LOW": "FALSE"}
    verdict = verdict_map.get(verdict_line.upper(), "UNCERTAIN")

    return jsonify({
        "verdict": verdict,
        "ml_prediction": ml_pred,
        "ml_confidence": round(ml_conf, 3),
        "ml_label": "FAKE" if ml_pred == 1 else "REAL" if ml_pred == 0 else "UNKNOWN",
        "summary": summary.strip(),
        "analysis": analysis.strip(),
        "risks": risk_list,
        "verification_results": [],
        "claims_count": 0
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
