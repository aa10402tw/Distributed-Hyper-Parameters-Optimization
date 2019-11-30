import copy
import random

# === Random Variable === #
class CRV():
    def __init__(self, low, high, value=None, name=""):
        self.low  = low
        self.high = high
        self.name = name
        self.value = value
        if self.value is None:
            self.initValue()

    def initValue(self):
        self.value = random.uniform(self.low, self.high)

    def __str__(self):
        s = "{} = {} (Continuous Random Variable, low={}, high={})".format(
            self.name, self.value, self.low, self.high
        )
        return s

    def __eq__(self, other):
        return (self.__class__ == other.__class__
            and self.value == other.value)

    def __repr__(self):
        s = "{}={:.4f}".format(
            self.name, self.value
        )
        return s
    def __hash__(self):
        return hash(self.value)

    def copy(self):
        return copy.deepcopy(self)

class DRV:
    def __init__(self, choices, value=None, name=""):
        self.choices = choices
        self.name = name
        self.value = value
        if self.value is None:
            self.initValue()

    def initValue(self):
        self.value = random.sample(self.choices, k=1)[0]

    def __str__(self):
        s = "{} = {} (Discrete Random Variable, choices={})".format(
            self.name, self.value, self.choices
        )
        return s

    def __repr__(self):
        s = "{}={}".format(
            self.name, self.value
        )
        return s

    def __eq__(self, other):
        return (self.__class__ == other.__class__
            and self.value == other.value)

    def __hash__(self):
        return hash(self.value)

    def copy(self):
        return copy.deepcopy(self)


class HyperParams:
    def __init__(self, rvs):
        self.rvs = rvs

    def initValue(self):
        for i in range(len(self.rvs)):
            self.rvs[i].initValue()
        return self

    def getValueTuple(self):
        return tuple([rv.value for rv in self.rvs])

    def __str__(self):
        s = "[Hyper-Parameters]\n"
        for i, rv in enumerate(self.rvs):
            s += "  var {} : {}\n".format(i+1, rv)
        return s

    def __repr__(self):
        s = "(HyperParams:"
        for i, rv in enumerate(self.rvs):
            s += "{},".format(rv.__repr__())
        s += ")\n"
        return s

    def __len__(self):
        return len(self.rvs)

    def __getitem__(self, key):
        return self.rvs[key]

    def __setitem__(self, key, value):
        self.rvs[key] = value

    def __hash__(self):
        return hash( self.__repr__() )

    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        if not len(self) == len(other):
            return False
        for (rv_1, rv_2) in zip(self.rvs, other.rvs):
            if rv_1 != rv_2:
                return False
        return True

    def copy(self):
        return copy.deepcopy(self)


def test_hparams():
    lr = CRV(low=0.0, high=1.0, name="lr")
    dr = CRV(low=0.0, high=1.0, name="dr")
    bs = DRV(choices=[16, 32, 64, 128, 256], name="bs")
    hparams = HyperParams([lr, dr, bs], )
    print(hparams)

if __name__ == "__main__":
    test_hparams()