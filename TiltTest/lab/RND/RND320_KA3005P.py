from ..generic import InterfaceSerial

class RND320_KA3005P(InterfaceSerial):

    # Manual here: https://media.distrelec.com/Web/Downloads/_m/an/RND_320_KA3000_mul_man.pdf
    # Remote command manual: https://media.distrelec.com/Web/Downloads/_t/ds/RND_320-KA_Control_Commands_eng_tds.pdf

    def __init__(self, port, log, baudrate=9600):
        super().__init__(port, baudrate, log=log)

    def SetCurrent(self, current):
        """Set the output current limit (`current` in A)."""
        self.query(f"ISET1:{float(current)}")

    def GetCurrent(self):
        """Returns the current settting."""
        return float(self.query(f"ISET1?", resp=True))

    def MeasureCurrent(self):
        """Returns the measured output current."""
        return float(self.query(f"IOUT1?", resp=True))

    def SetVoltage(self, voltage):
        """Set the output voltage limit (`voltage` in V)."""
        self.query(f"VSET1:{float(voltage)}")

    def GetVoltage(self):
        """Returns the voltage settting."""
        return float(self.query(f"VSET1?", resp=True))

    def MeasureVoltage(self):
        """Returns the measured output voltage."""
        return float(self.query(f"VOUT1?", resp=True))

    def On(self):
        """Turn the output on."""
        self.query("OUT1")

    def Off(self):
        """Turn the output off."""
        self.query("OUT0")

    def Status(self):
        """Returns True when the output is on, False otherwise."""

        # Bit 6: Output ON=1, Off=0
        # Convert the read ASCII value to an int with ord()
        return (ord(self.query("STATUS?", resp=True)) & (1 << 6)) != 0
