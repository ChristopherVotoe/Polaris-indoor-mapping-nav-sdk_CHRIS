from flask import *
from fileinput import filename
#from distutils.log import debug
import numpy as np
import ezdxf as ez
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'dxf'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'tmp'


def allowed_file(filename):
  return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_points_from_start_to_end(e, standard_point_scale):
  st = np.array([e.dxf.start.x, e.dxf.start.y, 0])
  ed = np.array([e.dxf.end.x, e.dxf.end.y, 0])
  st2 = np.array([e.dxf.start.x, e.dxf.start.y, 131.23])
  ed2 = np.array([e.dxf.end.x, e.dxf.end.y, 131.23])
  st3 = np.array([e.dxf.start.x, e.dxf.start.y, 78.74])
  ed3 = np.array([e.dxf.end.x, e.dxf.end.y, 78.74])

  dist_btw = np.linalg.norm(ed - st)
  dist_btw2 = np.linalg.norm(ed2 - st2)
  dist_btw3 = np.linalg.norm(ed3 - st3)


  num_points = int(dist_btw / standard_point_scale)
  num_points2 = int(dist_btw2 / standard_point_scale)
  num_points3 = int(dist_btw3 / standard_point_scale)

  # Generate intermediate points
  points = [st + i * (ed - st) / num_points for i in range(num_points)]
  points += [st2 + i * (ed2 - st2) / num_points2 for i in range(num_points2)]
  points += [st3 + i * (ed3 - st3) / num_points3 for i in range(num_points3)]
  points.append(ed)  # Add end point
  points.append(ed2)
  points.append(ed3)
  return points


def get_midpoint(e):
  st = np.array([e.dxf.start.x, e.dxf.start.y, 0])
  ed = np.array([e.dxf.end.x, e.dxf.end.y, 0])
  return (st + ed) / 2


def generate_point_cloud(dxfFile, name):
    try:
        doc = ez.readfile(dxfFile)
    except IOError:
        print("Not a DXF file or a generic I/O error.")
        return
    except ez.DXFStructureError:
        print("Invalid or corrupted DXF file.")
        return

    msp = doc.modelspace()
    standard_point_scale = 30

    cloud_point_array = []


    for e in msp.query('LINE[layer=="A-WALL"]'):
        cloud_point_array.extend(
            get_points_from_start_to_end(e, standard_point_scale)
        )


    door_points = []
    for e in msp.query('LINE[layer=="A-DOOR"]'):
        mid = get_midpoint(e)
        mid[2] = 300 #Door location on map
        door_points.append(mid)


    for e in msp.query('INSERT'):
        if e.dxf.layer == "A-DOOR":
            x = e.dxf.insert.x
            y = e.dxf.insert.y
            door_points.append(np.array([x, y, 300.0]))


    stairs_points = []
    for e in msp.query('LINE[layer=="A-FLOR-STRS"]'):
        mid = get_midpoint(e)
        mid[2] = 250  #Stair location on map
        stairs_points.append(mid)

    # merge everything
    cloud_point_array.extend(door_points)
    cloud_point_array.extend(stairs_points)

    if not cloud_point_array:
        print("No geometry found in expected layers.")
        return

    # to numpy + meters
    pointCloud = np.asarray(cloud_point_array, dtype=float)
    pointCloudMeters = (pointCloud * 0.0254)

    df = pd.DataFrame(pointCloudMeters, columns=['X', 'Y', 'Z'])


    x_min = df['X'].min()
    y_min = df['Y'].min()
    df['X'] -= x_min
    df['Y'] -= y_min

    pointCloudName = name + ".xyz"
    np.savetxt(os.path.join("tmp", pointCloudName), pointCloudMeters)

    # visualize
    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Z', title=name)
    fig.write_html(os.path.join("templates", "pc.html"))

    return pointCloudName




@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('success', name=filename))
    return render_template("index.html")


@app.route("/success/<name>")
def success(name):
    file = os.path.join(app.config['UPLOAD_FOLDER'], name)
    base, ext = os.path.splitext(name)
    pointCloudName = generate_point_cloud(file, base)

    # read the generated plotly html
    with open(os.path.join("templates", "pc.html"), "r", encoding="utf-8") as f:
        pc_html = f.read()

    return render_template(
        "Acknowledgement.html",
        name=name,
        pointCloudName=pointCloudName,
        pc_html=pc_html
    )


@app.route("/download/<pointCloudName>")
def download(pointCloudName):

    @after_this_request
    def clear_files(response):
        base, ext = os.path.splitext(pointCloudName)
        xyz_path = os.path.join(app.config['UPLOAD_FOLDER'], base + ".xyz")
        dxf_path = os.path.join(app.config['UPLOAD_FOLDER'], base + ".dxf")

        # delete only if exists
        if os.path.exists(xyz_path):
            os.remove(xyz_path)
        if os.path.exists(dxf_path):
            os.remove(dxf_path)

        return response

    return send_from_directory(app.config['UPLOAD_FOLDER'], pointCloudName)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)