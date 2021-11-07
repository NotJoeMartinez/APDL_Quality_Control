import uuid
import datetime as dt
from flask import Flask, request, jsonify
from PIL import Image
from .predictions import predict


app = Flask(__name__)

@app.route("/get-labels", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    predictions_dict = predict.make_predictions(img)

    file_name = f"APDL_QC_API/imgs/{str(uuid.uuid1())}.jpg"
    img.save(file_name)



    return jsonify({
                    'Predicted Label': predictions_dict['Predicted Label'], 
                    'Prediction Confidence': predictions_dict['Prediction Confidence'],
                    'All Predictions': predictions_dict['All Predictions'],
                    'Image Size': [img.width, img.height]})



    




if __name__ == "__main__":
    app.run(debug=True)