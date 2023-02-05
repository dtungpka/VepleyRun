#create a udp client and send HI to localhost port 3939
import socket
import time
import sys

UDP_IP  = "127.0.0.1"
UDP_PORT = 3939
MESSAGE = "STOP"
bytesToSend = str.encode(MESSAGE)
buffer_size = 1024

UDPClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
UDPClient.sendto(bytesToSend, (UDP_IP, UDP_PORT))
print("Message sent to server")
received = UDPClient.recvfrom(buffer_size)
print("Received message: ", received[0].decode())
