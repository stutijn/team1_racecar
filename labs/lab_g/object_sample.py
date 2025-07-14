
class AIRadio(Radio):
    def turn_on(self):
        pass

class TouchRadio (Radio):
    def turn_on(self):
        pass

class Radio(object):
    def turn_on(self):
        pass

    def turn_off(self):
        pass

    def change_station(self, staion_code):
        pass

def main():
    my_radio = TouchRadio()
    my_radio.turn_on()

    ai_radio = AIRadio()
    ai_radio.turn_on()