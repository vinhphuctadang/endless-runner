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

## Socket connect:


### Server side script:

```
import socket_helper as sock
from threading import Thread 
Thread(target=sock.start_listening,).start()
```

### Unity script

```
using System;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Linq;

public class SocketListener : MonoBehaviour
{
    // Start is called before the first frame update
    private Socket sock;
    private static ManualResetEvent connectDone =
        new ManualResetEvent(false);  
    static string decode_message(byte[] raw){
        string result = "";
        for(int i = 0; i<raw.Length && raw[i] != 0; ++i) {
            result += (char) raw[i];
        }
        return result;
    }

    IEnumerator listenSocket(){
        
        while(true) {
            // read data if available
            if (sock.Available > 0) {
                // reset buffer
                var buffer = new byte[128];

                // receive data from socket
                var readCount = sock.Receive(buffer);

                if (readCount == 0) {
                    // stop the corroutine
                    yield return null;
                }

                // decode it
                var message = decode_message(buffer);
                // return to caller
                Debug.Log(message);
                yield return message;
            }
            yield return "";
        }
    }

    void Start()
    {
        // connect 
        sock = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        sock.Blocking = false;
        sock.ReceiveTimeout = 1000; // 100 ms for receiving
        // connect async
        sock.BeginConnect(new IPEndPoint(IPAddress.Parse("127.0.0.1"), 5000),
            new AsyncCallback(ConnectCallback), sock);
        connectDone.WaitOne();
        // now start corroutine
        StartCoroutine("listenSocket");
        // cannot start corroutine in another thread, which prevent corroutine to work with unity main thread
    }

    private void ConnectCallback(IAsyncResult ar) {
        // after connecting done call to start corroutine
        // Retrieve the socket from the state object.  
        sock = (Socket) ar.AsyncState;  
        // Complete the connection phase
        sock.EndConnect(ar);
        connectDone.Set();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

```