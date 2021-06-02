# Endless runner game
---

## What we are doing?

We build a game controlled using human pose

## What to run ?

Run ``pose/server/app.py`` for starting server, camera activation needed

--- 

## Folder structure inside /pose

### For action classification
Files in this folder will be operated in such a way that every folder's name forms a label, that folder contains it video sample in which there are repeated motions of the action, in .mov or .mp4 format

```
| -- action 1
| -- -- 1_1.mov
| -- -- 1_2.mov
| -- -- 1_3.mov
| -- action 2
| -- -- 2_1.mov
| -- -- 2_2.mov
| ..
| -- action n
| -- -- n_1.mov
| -- -- n_2.mov
```

/extract_features_action_research.ipynb will read and extract features from the video, whereas /train_action.ipynb will read extracted feature and train with LSTM

dcongtinh is in charge of maintaining this folder

### For pose classification

Training to recognize posture is easier than to recognize actions, thus the folder will only contain .mov/.mp4 files and all we need is to modify /extract_video_feature_posture.py to extract features and train using train_posture.ipynb

```
| posture_1_1.mov
| posture_2_1.mov
| posture_2_2.mov
| ...
| posture_n_k.mov
```

In each video, creator must ensure that characters posture is maintained (slightly different allowed!) to make a clean data folder. Lighting condition should be good, too.

## Web socket usage:

- Unity: 

```https://github.com/nhn/socket.io-client-unity3d/releases/tag/v.1.1.2```

**Note**: Require socket.io protocol revision 3, 4

- Python server side requirements (must):

```
flask-socketio==4
python-engineio==3.2.0
python-socketio==3.0.0
```

### Unity C# socket io usage implementation example:

```
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using socket.io;

public class SocketListener : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        var socket = Socket.Connect("http://localhost:5000");

        // listen on event, may push into queue
        socket.On("ping", (string response) =>
        {
            UnityEngine.Debug.Log($"server: {response}");
        });
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
```