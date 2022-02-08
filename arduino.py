from pyfirmata import Arduino, SERVO

try:
    port='COM5'
    pin=3

    r=11
    g=10
    b=9
    buzz=6

    board=Arduino(port)
    board.digital[pin].mode=SERVO
    
    board.digital[r].write(1)
    board.digital[g].write(1)
    board.digital[b].write(1)
    board.digital[buzz].write(0)
            


except:
    print("arduino not connected")


def rotateservo(pin,angle):
    board.digital[pin].write(angle)

def buzzer(val):
    try:
        board.digital[buzz].write(val)
    except:
        pass

def color(rd,gr,bl):
    try:
        board.digital[r].write(rd)
        board.digital[g].write(gr)
        board.digital[b].write(bl)
    except:
        pass

def lock(val):
    try:
        if (val==0):
            rotateservo(pin, 70)
            color(1,0,1)
            
            
        elif (val==1):
            rotateservo(pin, 0)
            color(0,1,1)
    except:
        pass