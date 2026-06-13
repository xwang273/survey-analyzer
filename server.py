import io
import csv
import os
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

# HoughCircles tuning constants — adjust if detection is poor on your images
HC_DP = 1.2
HC_MIN_DIST = 20
HC_PARAM1 = 50       # upper Canny threshold
HC_PARAM2 = 30       # accumulator threshold (lower → more circles found)
HC_MIN_RADIUS = 8
HC_MAX_RADIUS = 40
Y_TOLERANCE = 20     # px — circles within this Y range share a row
INNER_RATIO = 0.6    # fraction of radius used for fill-intensity sampling


def detect_radio_buttons(image_bytes, min_val, max_val, num_options):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupted file.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HC_DP,
        minDist=HC_MIN_DIST,
        param1=HC_PARAM1,
        param2=HC_PARAM2,
        minRadius=HC_MIN_RADIUS,
        maxRadius=HC_MAX_RADIUS,
    )

    if circles is None:
        raise ValueError(
            "No circles detected. Try adjusting the image (better contrast, less skew) "
            "or ensure the survey has standard radio buttons."
        )

    circles = np.round(circles[0]).astype(int)

    # Sort all circles by Y then cluster into rows
    circles = circles[circles[:, 1].argsort()]
    rows = []
    current_row = [circles[0]]
    for c in circles[1:]:
        row_mean_y = int(np.mean([r[1] for r in current_row]))
        if abs(int(c[1]) - row_mean_y) <= Y_TOLERANCE:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)

    if len(rows) < 1:
        raise ValueError("No rows detected in image.")

    result_rows = []
    for row_idx, row in enumerate(rows):
        # Sort left → right
        row_sorted = sorted(row, key=lambda c: c[0])

        darkest_pos = 0
        darkest_mean = float("inf")
        for pos_idx, (cx, cy, radius) in enumerate(row_sorted):
            inner_r = max(1, int(radius * INNER_RATIO))
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), inner_r, 255, thickness=-1)
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            if mean_intensity < darkest_mean:
                darkest_mean = mean_intensity
                darkest_pos = pos_idx

        if num_options == 1:
            value = float(min_val)
        else:
            value = min_val + (darkest_pos / (num_options - 1)) * (max_val - min_val)

        result_rows.append({
            "row": row_idx + 1,
            "position": darkest_pos,
            "value": round(value, 4),
        })

    if num_options == 1:
        scale_labels = [float(min_val)]
    else:
        scale_labels = [
            round(min_val + (i / (num_options - 1)) * (max_val - min_val), 4)
            for i in range(num_options)
        ]

    return {"rows": result_rows, "scale_labels": scale_labels}


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        file = request.files.get("image")
        if file is None or file.filename == "":
            return jsonify({"error": "No image uploaded."}), 400

        try:
            min_val = float(request.form.get("min_val", 1))
            max_val = float(request.form.get("max_val", 5))
            num_options = int(request.form.get("num_options", 5))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid scale configuration values."}), 400

        if num_options < 1:
            return jsonify({"error": "Num Options must be at least 1."}), 400
        if min_val > max_val:
            return jsonify({"error": "Min value must be less than or equal to Max value."}), 400

        image_bytes = file.read()
        result = detect_radio_buttons(image_bytes, min_val, max_val, num_options)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/export", methods=["POST"])
def export():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON body."}), 400

        rows = data.get("rows", [])
        if not rows:
            return jsonify({"error": "No rows to export."}), 400

        values = [r["value"] for r in rows if r.get("value") is not None]
        average = round(sum(values) / len(values), 4) if values else ""

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Row", "Value"])
        for r in rows:
            writer.writerow([r["row"], r["value"]])
        writer.writerow([])
        writer.writerow(["Average", average])

        output.seek(0)
        bytes_output = io.BytesIO(output.getvalue().encode("utf-8"))

        return send_file(
            bytes_output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="survey_results.csv",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
