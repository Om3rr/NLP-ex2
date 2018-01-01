class EZSparseVector:
    def __init__(self, values, positions):
        if(len(values) != len(positions)):
            raise ValueError("positions length should be like values length")
        self.__vals = {positions[i] : values[i] for i in range(len(values))}

    def __getitem__(self, item):
        if item in self.__vals:
            return self.__vals[item]
        else:
            return 0

    def __setitem__(self, key, value):
        if value == 0:
            if self[key] == 0:
                return
            else:
                del self.__vals[key]
        else:
            self.__vals[key] = value
# assuming that the other is the smaller one.
    def dot(self, other):
        sum = 0
        for pos in other.__vals.keys():
            sum += self[pos] * other[pos]
        return sum

    def __sub__(self, other):
        for pos, value in other.__vals.items():
            self[pos] -= value
        return self

    def __add__(self, other):
        for pos, value in other.__vals.items():
            self[pos] += value
        return self

    def __truediv__(self, other):
        if(type(other) in [float, int]):
            for pos in self.__vals:
                self[pos] /= other
            return self
        else:
            raise Exception("Unsupported operation")

    def __mul__(self, other):
        for pos in self.__vals:
            self[pos] *= other
        return self
    def __str__(self):
        return str(self.__vals.items())

    def copy(self):
        return EZSparseVector(list(self.__vals.values()), list(self.__vals.keys()))

    def items(self):
        return self.__vals.items()
