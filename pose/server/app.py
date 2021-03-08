from flask import Flask
app = Flask(__name__)

global device = None

@app.route("/")
def ping():
    return {"code": 1}

@app.route("/pose")
def get_pose():
    return {"code": 1, "pose": []}

@app.route("/init")
def get_pose():
    try:
        device = cv2.VideoCapture(VIDEO_URI)
    except Exception as e:
        return {"code": -1, "message": str(e)}
    return {"code": 1}

@app.route("/done")
def get_pose():
    return {"code": 1}

if __name__=="__main__":
    app.run(debug=True)