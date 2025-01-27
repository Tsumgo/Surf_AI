import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

keyList.extend([0x25, 0x26, 0x27, 0x28]) 

def key_check():
    keys = []
    for key in keyList:
        # if wapi.GetAsyncKeyState(ord(key)):
        if wapi.GetAsyncKeyState(ord(key) if isinstance(key, str) else key): 
            keys.append(key)



    if 'H' in keys: 
        return 'H'
    elif ' ' in keys:
        return ' '
    elif 0x26 in keys: 
        return 'Up'
    elif 0x28 in keys: 
        return 'Down'
    elif 0x25 in keys:
        return 'Left'
    elif 0x27 in keys:
        return 'Right'
    elif 'A' in keys:
        return 'A'
    elif 'D' in keys:
        return 'D'
    elif 'S' in keys:
        return 'S'
    elif 'B' in keys:
        return 'B'
    else:
        return 'Q'
