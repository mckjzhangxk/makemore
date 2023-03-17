import socket

localIP = "0.0.0.0"

localPort = 8888

bufferSize = 1024

msgFromServer = "Hello UDP Client"

bytesToSend = str.encode(msgFromServer)

# Create a datagram socket

UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip

UDPServerSocket.bind((localIP, localPort))

while (True):
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)

    message = bytesAddressPair[0]

    address = bytesAddressPair[1]

    clientMsg = "Message from Client"


    print(message)
    print(address)

    # Sending a reply to client
    clientIp="%s:%d"%(address[0],address[1])
    UDPServerSocket.sendto(str.encode(clientIp), address)