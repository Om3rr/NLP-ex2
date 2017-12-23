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

    def dot(self, other):
        a,b = set(self.__vals.keys()), set(other.__vals.keys())
        poss = a.intersection(b)
        sum = 0
        for pos in poss:
            sum += self[pos] * other[pos]
        return sum

    def __sub__(self, other):
        for pos, value in other.__vals.items():
            self[pos] -= value
        return self

    def __str__(self):
        return str(self.__vals.items())


