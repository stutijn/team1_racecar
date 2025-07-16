BLUE = ((90, 50, 50), (120, 255, 255))  # The HSV range for the color blue
GREEN = ((35,100,50),(85,255,255))  # The HSV range for the color green
RED = ((170,50,50),(10,255,255))  # The HSV range for the color red
ORANGE = ((10,100,100),(25,255,255)) # The HSV range for the color orange
YELLOW = ((10,100,50),(35,255,255)) # The HSV range for the color yellow
PURPLE = ((140,50,50),(160,255,255)) # The HSV range for the color purple


color_set = [(BLUE,"blue"), GREEN, RED, ORANGE, YELLOW, PURPLE]

def use():
    for color in color_set:
        if color != 0:
            color = color_set[0][1]
        
    return(str(color))



utility = use()
if use == 0:
    print("0")

else:
    print(utility)