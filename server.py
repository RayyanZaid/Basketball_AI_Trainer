import os
from flask import Flask, request,abort,jsonify


app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.mp4','.MOV']



@app.route('/' , methods=['GET', 'POST'])
def welcome():
    return "<h2> Hello and welcome to my basketball project </h2>"



@app.route('/analize' , methods = ['POST'])
def compareVideos():
    uploaded_video1 = request.files.getlist("video1")[0]
    video1Name: str = uploaded_video1.filename


    uploaded_video2 = request.files.getlist("video2")[0]
    video2Name: str = uploaded_video2.filename
    # Checking 

    # 1) If there is a video 


    if video1Name != "" and video2Name != "":
        _, video_file_ext1 = os.path.splitext(video1Name)
        _, video_file_ext2 = os.path.splitext(video2Name)


        # if either extension is not in the extensions, then we abort                                
        if video_file_ext1 not in app.config['UPLOAD_EXTENSIONS'] or video_file_ext2 not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)


        # Now, the videos are validated

        
        uploaded_video1.save(video1Name)
        uploaded_video2.save(video2Name)






if __name__ == "__main__":
    app.run()