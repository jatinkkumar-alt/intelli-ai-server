import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
# -------------------------
# CONFIG
# -------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "models/gemini-flash-latest"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

@app.route("/ask-ai", methods=["POST"])
def ask_ai():
    """
    Request JSON:
    {
        "question": "string",
        "prediction": { ... },   # lastPrediction from frontend (optional)
        "symptoms": ["fever", "cough", ...]
    }
    Response JSON:
    { "answer": "text..." } or { "error": "..." }
    """
    data = request.get_json() or {}
    user_question = (data.get("question") or "").strip()
    prediction = data.get("prediction") or {}
    symptoms = data.get("symptoms") or []

    # If prediction info is missing but we have symptoms, recompute:
    if (not prediction) and symptoms:
        prediction = make_prediction(symptoms)

    disease = prediction.get("disease", "Unknown")
    confidence = prediction.get("confidence", None)
    top_predictions = prediction.get("top_predictions", [])

    # Build prompt (general info, NOT diagnosis)
    lines = []
    lines.append("You are an AI medical information assistant.")
    lines.append("You must NOT give a confirmed diagnosis or prescribe medicine.")
    lines.append("Always encourage the user to consult a doctor for actual medical decisions.")
    lines.append("")
    lines.append("Here is an AI model based disease risk assessment:")
    lines.append(f"- Predicted disease: {disease}")
    if confidence is not None:
        lines.append(f"- Model confidence: {confidence*100:.2f}%")
    if top_predictions:
        lines.append("- Top possible conditions:")
        for p in top_predictions[:3]:
            conf_str = f"{p['confidence']*100:.2f}%" if p.get("confidence") is not None else "N/A"
            lines.append(f"  - {p['disease']} ({conf_str})")
    if symptoms:
        lines.append(f"- Reported symptoms: {', '.join(symptoms)}")
    lines.append("")
    lines.append("User question:")
    lines.append(user_question or "Explain this result in simple language.")
    lines.append("")
    lines.append("Answer in 120-160 words, in simple, calm language.")
    lines.append("Include only general information and lifestyle guidance.")
    lines.append("Do NOT sound like you are confirming a diagnosis.")
    lines.append("End by reminding them to visit a doctor for confirmation.")

    prompt = "\n".join(lines)

    try:
        # Adjust payload format based on how YOUR Render API expects it
        payload = {"prompt": prompt}

        resp = requests.post(GEMINI_API_URL, json=payload, timeout=20)
        resp.raise_for_status()
        resp_json = resp.json()

        # Adjust this based on your Render response structure
        answer = resp_json.get("response", str(resp_json))


        return jsonify({"answer": answer})

    except Exception as e:
        print("Error calling Gemini API:", e)
        return jsonify({"error": "AI assistant is not available right now."}), 500

# -------------------------
# ROUTE: AI ASSISTANT
# -------------------------

@app.route("/intelli-ai", methods=["POST"])
def intelli_ai():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        history = data.get("history", "").strip()

        if not user_message:
            return jsonify({"reply": "No message received from app."})

        print(f"[AI] Incoming from app: {user_message}")
        print(f"[AI] History length: {len(history)} characters")

        prompt = f"""
        You are Intelli, the AI assistant inside a messaging app called IntelliChat.
        You must answer based on the conversation history and the latest user message. 
        Remember the details told by User which can help to chat with user in continuity regarding a certain topic.

        Here is the conversation so far (from oldest to newest):
        {history}

        Now the user says: {user_message}

        Respond as Intelli in a friendly, helpful, and concise way. Do not repeat the full history.
        """

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        reply_text = getattr(response, "text", None)
        if not reply_text:
            if hasattr(response, "candidates") and response.candidates:
                try:
                    reply_text = response.candidates[0].content.parts[0].text
                except Exception:
                    reply_text = "AI had no response."
            else:
                reply_text = "AI had no response."

        print(f"[AI] Reply: {reply_text[:80]}...")
        return jsonify({"reply": reply_text})

    except Exception as e:
        print(f"[AI] ERROR: {e}")
        return jsonify({"reply": f"AI error: {str(e)}"}), 500

import json
...

@app.route("/smart-reply", methods=["POST"])
def smart_reply():
    try:
        data = request.get_json()
        history = data.get("history", "").strip()
        last_message = data.get("lastMessage", "").strip()

        if not last_message:
            return jsonify({"suggestions": []})

        print(f"[SMART-REPLY] Last message: {last_message}")
        print(f"[SMART-REPLY] History length: {len(history)} chars")

        prompt = f"""
You are Intelli, an AI assistant inside the IntelliChat app.

You will be given a short conversation history and the latest incoming message.
Your job is to suggest 3 short, natural, and human-like reply options that the user could send next.

Rules:
- Each suggestion should be SHORT (max 1–2 sentences).
- Tone should be friendly and natural, no emojis unless clearly appropriate.
- Don't repeat exactly what the other person said.
- Adapt to the context in the history.

Conversation history:
{history}

Latest incoming message (the other person just sent this):
{last_message}

Now respond with ONLY a valid JSON array of 3 strings.
Example:
["Sure, I can help with that.", "Can you share more details?", "Let's start step by step."]

Do NOT add any extra text outside the JSON.
"""

        if not API_KEY:
            return jsonify({"suggestions": ["(Server misconfigured: no API key)"]})

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        raw_text = getattr(response, "text", "") or ""
        raw_text = raw_text.strip()
        print(f"[SMART-REPLY] Raw model output: {raw_text}")

        suggestions = []

        # Try to parse as JSON array
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                suggestions = [str(s).strip() for s in parsed if str(s).strip()]
        except Exception as e:
            print(f"[SMART-REPLY] JSON parse failed: {e}")
            # Fallback: split by lines
            for line in raw_text.splitlines():
                line = line.strip().lstrip("-•1234567890. ").strip()
                if line:
                    suggestions.append(line)
                if len(suggestions) == 3:
                    break

        # Hard fallback if model misbehaves
        if not suggestions:
            suggestions = [
                "Got it, tell me more.",
                "That sounds interesting.",
                "Can you explain a bit more?"
            ]

        print(f"[SMART-REPLY] Final suggestions: {suggestions}")
        return jsonify({"suggestions": suggestions})

    except Exception as e:
        print(f"[SMART-REPLY] ERROR: {e}")
        return jsonify({"suggestions": []}), 500

# -------------------------
# HEALTH CHECK ROUTE
# -------------------------

@app.route("/", methods=["GET"])
def health():
    return "Intelli AI Server is running.", 200


# -------------------------
# ENTRY POINT
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running locally on port {port}")
    app.run(host="0.0.0.0", port=port)
