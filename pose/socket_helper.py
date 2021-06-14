import socket               # Import socket module
import time

need_disconnect = False
socket_instance = None
connected_clients = set()

# [total_frame_size: 4 byte] [[len_i: 4 byte][param_i: len_i byte]]
def encode_message(msg):
    # pack a message to 128-byte frame
    byte_msg = bytes(msg, encoding='utf8')
    message = byte_msg + b'\1' * (128 - len(byte_msg))
    return message

def broadcast(message):
    global connected_clients
    print("Connected client size:", len(connected_clients))
    message = bytes(message, encoding="utf8")

    remove_clients = []
    for client in connected_clients:
        try:
            byte_sent = client.send(message)
            print("Byte sent:", byte_sent)
        except Exception as e:
            print("Error happened:", e)
            remove_clients.append(client)
    for client in remove_clients:
        connected_clients.remove(client)

def start_listening(HOST="0.0.0.0", PORT=5000):
    global socket_instance
    global connected_clients
    global need_disconnect
    # new and config socket
    socket_instance = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_instance.setblocking(False)
    socket_instance.settimeout(600)
    need_disconnect = False
    config = (HOST, PORT)
    print("[status] socket is starting listening on ", config)
    socket_instance.bind(config)
    socket_instance.listen(8)
    connected_clients = set()
    # start listening for new connections
    try:
        while not need_disconnect:
            print("[status] listening for new clients")
            try:
                client, addr = socket_instance.accept()     # Establish connection with client.
                print('[status] got connection from', addr)
                connected_clients.add(client)
                # broadcast(encode_message("welcomeEvent " + str(addr)))
            except Exception as e:
                print("Exception:", e)
                time.sleep(1)
    except KeyboardInterrupt:
        print("^C interrupted")
    socket_instance.close()
    print("[status] socket disconnected")

def stop_listening():
    global need_disconnect
    need_disconnect = True
